"""Soft-loss test (2026-10-23): Studio · multi-scroll continued with Youssef v8in's loss instead of Studio's.

Question: is Studio's lack of soft, diffuse responses (useful for iterative human labelling) mainly its loss? Everything is
the multi-scroll CPT run's (studio-multiscroll-cpt-20261023/train_cpt.py: data, replay, mixture, augmentation, model,
128 px window, evaluation) except
  * start: that run's published EMA weights (model-ema.safetensors, SHA-256 ec1f239a...), fresh AdamW;
  * loss: BCE with label smoothing SOFT_SMOOTH (0.25: targets 0.25 background / 0.75 ink, as
    smp.losses.SoftBCEWithLogitsLoss(smooth_factor=0.25) in YoussefMoNader/ink-8um-v8in) + Dice computed over the whole
    microbatch with hard targets and smooth 0, skipped when the microbatch has no labeled ink (smp DiceLoss(mode='binary'),
    v8in loss_mode='batch'), 0.5 each. BCE stays Studio's per-example masked mean over the central 96 px averaged over
    contributing examples; unknown pixels stay out of both terms;
  * schedule: 3,000 updates, 200-update warm-up, peak 6e-5, cosine to 2 %, EMA 0.998; evaluation every 1,000.
Below, the parent run's description (unchanged parts).

Start (parent run): the published Studio epoch-9 weights (DeepResidualUNet(deeper=False), 223,355,921 parameters; fresh AdamW).
Every optimizer update (64 examples, 4 microbatches of 16, exactly Studio's) holds
  * REPLAY examples from Studio's own sparse-label data, PHerc0814 / PHerc0841 / PHerc0009B in equal shares
    (Studio's scroll-balanced epoch order, consumed continuously), from Studio's GPU-resident tile store; and
  * the rest from other scrolls (host-RAM atlas, Expanded9 prepared-row format), drawn per update from scroll groups
    by the weights in mixture.json (groups without data are skipped and the weights renormalized).
Studio's coupled augmentation (`sparse_data.augment` / `parameters`, scroll-shift-v2) and loss (per-example masked
0.5 BCE + 0.5 Dice on the central 96 px, averaged over contributing examples) are reused unchanged; dense labels from
other sources become Studio's categorical state (multiscroll_data.categorical).
Evaluation every EVAL_EVERY updates: held-out PHerc0211 AP (eval211.py; never trained on) and Studio's fixed 841 holdout.

Environment: CPT_TOTAL_UPDATES (30000), CPT_PEAK_LR (6e-5; this run used 1e-4 for updates 0-2,4xx, see lr_history), CPT_WARMUP (600), CPT_REPLAY (20 of 64), CPT_EVAL_EVERY (2000),
STUDIO_WANDB=1 for W&B, STUDIO_GPU_LEASE=0 to skip the GH200 shared-GPU lease.
"""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1'); os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '16'); os.environ.setdefault('PYTORCH_ALLOC_CONF', 'expandable_segments:True')
import concurrent.futures as cf, copy, hashlib, json, math, signal, sys, time
from collections import deque, defaultdict
from pathlib import Path
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
STUDIO = ROOT.parent / 'donor-studio-resunet-scratch-20260926'
CPT = ROOT.parent / 'studio-multiscroll-cpt-20261023'  # parent run: mixture.json, eval211, multiscroll_data, gpu_lease
sys.path.insert(0, str(STUDIO)); sys.path.insert(0, str(CPT)); sys.path.insert(0, str(ROOT))
from studio_env import load_env, localize
load_env()
from sparse_data import augment, parameters
from small_unets import DeepResidualUNet
from multiscroll_data import HostAtlas, GroupStream
from eval211 import Eval211
from gpu_lease import acquire
sys.path.insert(0, str(ROOT.parent))
from reproduction import validate_rows

RUN_ID = 'studio-softloss-20261023'
TOTAL = int(os.environ.get('CPT_TOTAL_UPDATES', '3000')); PEAK = float(os.environ.get('CPT_PEAK_LR', '6e-5'))
WARMUP = int(os.environ.get('CPT_WARMUP', '200')); REPLAY = int(os.environ.get('CPT_REPLAY', '20'))
EVAL_EVERY = int(os.environ.get('CPT_EVAL_EVERY', '1000')); FLOOR = .02
EMA_DECAY = float(os.environ.get('CPT_EMA_DECAY', '0.998'))
SMOOTH = float(os.environ.get('SOFT_SMOOTH', '0.25'))
STUDIO_SCROLLS = ('PHerc0814', 'PHerc0841', 'PHerc0009B')
STOP = False


def atomic(path, value):
    tmp = path.with_suffix('.tmp'); tmp.write_text(json.dumps(value, indent=2) + '\n'); tmp.replace(path)


def learning_rate(step):
    return PEAK * min(1., (step + 1) / WARMUP) * (FLOOR + (1 - FLOOR) * (1 + math.cos(math.pi * min(step, TOTAL) / TOTAL)) / 2)


def masked_terms_vector(logits, categorical):
    """Studio's sparse_data.masked_terms, returning the per-example (bce + dice) * .5 * active vector."""
    from torch.nn import functional as F
    z = logits[..., 16:112, 16:112].float(); state = categorical[..., 16:112, 16:112]
    y = (state == 2).float(); m = (state > 0).float(); axes = (1, 2, 3)
    count = m.sum(axes); active = count > 0
    bce = (m * F.binary_cross_entropy_with_logits(z, y, reduction='none')).sum(axes) / count.clamp_min(1)
    p = z.sigmoid(); dice = 1 - (2 * (m * p * y).sum(axes) + 1) / ((m * p).sum(axes) + (m * y).sum(axes) + 1)
    return (bce + dice) * .5 * active, active


def soft_terms(logits, categorical):
    """v8in-style loss on Studio's masked central 96 px.

    Returns (loss_sum, log_vector, active): loss_sum = 0.5 * sum of per-example masked soft-target BCE over contributing
    examples + 0.5 * microbatch Dice * number of contributing examples, so loss_sum / contributing (all four microbatches)
    keeps the parent run's 0.5 / 0.5 weighting. log_vector is the per-example 0.5 * (soft BCE + per-example Dice), only for
    the group-loss log (not optimized)."""
    from torch.nn import functional as F
    z = logits[..., 16:112, 16:112].float(); state = categorical[..., 16:112, 16:112]
    y = (state == 2).float(); m = (state > 0).float(); axes = (1, 2, 3)
    count = m.sum(axes); active = count > 0
    target = y * (1 - SMOOTH) + (1 - y) * SMOOTH
    bce = (m * F.binary_cross_entropy_with_logits(z, target, reduction='none')).sum(axes) / count.clamp_min(1)
    p = z.sigmoid(); mp = m * p; my = m * y
    positives = my.sum()
    dice_batch = (1 - 2 * (mp * y).sum() / (mp.sum() + positives).clamp_min(1e-7)) * (positives > 0)
    loss_sum = .5 * (bce * active).sum() + .5 * dice_batch * active.sum()
    dice_example = 1 - (2 * (mp * y).sum(axes) + 1) / (mp.sum(axes) + my.sum(axes) + 1)
    return loss_sum, ((bce + dice_example) * .5 * active).detach(), active


class StudioGpuData:
    """sparse_data.GpuData (Studio's shared 512 tiles resident in VRAM), dropping each file from the page cache after it
    is read so GH200's GPU NUMA node is not filled by cached tiles."""
    def __init__(self, doc):
        self.rows = [r for r in doc['segments'] if r['samples']]
        count = sum(r['tiles'] for r in self.rows); required = count * 512 * 512 * 17
        assert required + 24 * 2 ** 30 < torch.cuda.mem_get_info()[0], 'Studio tile store does not fit next to training'
        self.raw = torch.empty((count, 512, 512, 16), dtype=torch.uint8, device='cuda')
        self.state = torch.empty((count, 512, 512), dtype=torch.uint8, device='cuda')
        h = max((r['shape'][0] + 511) // 512 for r in self.rows); w = max((r['shape'][1] + 511) // 512 for r in self.rows)
        lookup = np.full((len(self.rows), h, w), -1, np.int32); self.coords = []
        entry_row = np.empty(doc['training_entries'], np.int32); entry_yx = np.empty((doc['training_entries'], 2), np.int32); start = 0
        for i, row in enumerate(self.rows):
            path = localize(row['folder']); tiles = np.load(path / 'tiles.npy'); values = np.load(path / 'ct-tiles.npy', mmap_mode='r')
            labels = np.load(path / 'labels.npy', mmap_mode='r'); coords = np.load(path / 'coordinates.npy'); self.coords.append(coords)
            for k, (y, x) in enumerate(tiles): lookup[i, y // 512, x // 512] = start + k
            for offset in range(0, len(tiles), 32):
                n = min(32, len(tiles) - offset)
                raw = torch.from_numpy(np.array(values[offset:offset + n], copy=True))
                categorical = np.zeros((n, 512, 512), np.uint8)
                for j, (y, x) in enumerate(tiles[offset:offset + n]):
                    block = labels[y:y + 512, x:x + 512]; categorical[j, :block.shape[0], :block.shape[1]] = block
                self.raw[start + offset:start + offset + n].copy_(raw); self.state[start + offset:start + offset + n].copy_(torch.from_numpy(categorical))
            for name in ('ct-tiles.npy', 'labels.npy'):
                fd = os.open(path / name, os.O_RDONLY); os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED); os.close(fd)
            if row['role'] == 'train':
                sl = slice(row['entry_offset'], row['entry_offset'] + row['samples']); entry_row[sl] = i; entry_yx[sl] = coords[np.load(path / 'entry-to-patch.npy')]
            start += len(tiles)
        self.lookup = torch.from_numpy(lookup).cuda(); self.entry_row = entry_row; self.entry_yx = entry_yx
        self.shapes = torch.tensor([r['shape'] for r in self.rows], device='cuda'); self.bytes = required
        print('STUDIO_GPU_TILES', count, 'tiles', round(required / 2 ** 30, 1), 'GiB', flush=True)

    def crops(self, indices, yx, width):
        indices = torch.as_tensor(indices, device='cuda', dtype=torch.long); yx = torch.as_tensor(yx, device='cuda', dtype=torch.long)
        ys = yx[:, 0, None] + torch.arange(width, device='cuda'); xs = yx[:, 1, None] + torch.arange(width, device='cuda')
        h, w = self.shapes[indices].unbind(1)
        outside = (ys[:, :, None] < 0) | (ys[:, :, None] >= h[:, None, None]) | (xs[:, None, :] < 0) | (xs[:, None, :] >= w[:, None, None])

        def reflect(a, n):
            a = a % (2 * n[:, None] - 2)
            return torch.where(a < n[:, None], a, 2 * n[:, None] - 2 - a)
        yy = reflect(ys, h); xx = reflect(xs, w)
        tile = self.lookup[indices[:, None, None], yy[:, :, None] // 512, xx[:, None, :] // 512].long()
        if bool((tile < 0).any()): raise RuntimeError('Unprepared input context tile')
        raw = self.raw[tile, yy[:, :, None] % 512, xx[:, None, :] % 512]
        state = self.state[tile, yy[:, :, None] % 512, xx[:, None, :] % 512].masked_fill_(outside, 0)
        return raw, state

    def fetch(self, entries, offsets):
        return self.crops(self.entry_row[entries], self.entry_yx[entries] - 112 + np.asarray(offsets), 224)

    def validation_coords(self):
        return self.coords[next(i for i, r in enumerate(self.rows) if r['role'] == 'validation')]

    def validation_batch(self, start, count=16):
        index = next(i for i, r in enumerate(self.rows) if r['role'] == 'validation'); coords = self.validation_coords()[start:start + count]
        return self.crops(np.full(len(coords), index), coords, 128)


def studio_epoch(doc, epoch):
    """Studio train.py's epoch_indices: scroll-balanced epoch order (seed 2026091201 + epoch)."""
    rng = np.random.default_rng(2026091201 + epoch); draws = []; count = math.ceil(max(doc['entry_counts'].values()) / 64) * 64
    for scroll in STUDIO_SCROLLS:
        pool = np.concatenate([np.arange(r['entry_offset'], r['entry_offset'] + r['samples']) for r in doc['segments'] if r['role'] == 'train' and r['samples'] and r['scroll'] == scroll])
        draws.append(np.concatenate([pool, rng.choice(pool, count - len(pool), replace=True)]))
    result = np.concatenate(draws); rng.shuffle(result); return result


class StudioStream:
    def __init__(self, doc):
        self.doc = doc; self.epoch = 0; self.position = 0; self.order = studio_epoch(doc, 0)

    def take(self, n):
        out = []
        while len(out) < n:
            if self.position == len(self.order):
                self.epoch += 1; self.position = 0; self.order = studio_epoch(self.doc, self.epoch)
            k = min(n - len(out), len(self.order) - self.position); out.extend(self.order[self.position:self.position + k]); self.position += k
        return np.array(out)

    def state(self): return dict(epoch=self.epoch, position=self.position)

    def load(self, s): self.epoch = s['epoch']; self.position = s['position']; self.order = studio_epoch(self.doc, self.epoch)


def drop_page_cache():
    """On GH200 the page cache can sit on the GPU's NUMA node and look like used GPU memory; flush it before loading
    (needs passwordless sudo; skipped silently otherwise or with CPT_DROP_CACHES=0)."""
    if os.environ.get('CPT_DROP_CACHES', '1') != '1': return
    import subprocess
    try:
        subprocess.run(['sync'], timeout=120); subprocess.run(['sudo', '-n', 'tee', '/proc/sys/vm/drop_caches'], input=b'1', capture_output=True, timeout=120)
    except Exception:
        pass


def main():
    drop_page_cache()
    mixture = json.loads((CPT / 'mixture.json').read_text())
    rows = [str((CPT / p).resolve()) for p in mixture['rows']]
    validate_rows(rows)
    doc = json.loads((STUDIO / 'data.json').read_text()); assert doc['status'] == 'complete'
    out = ROOT / 'checkpoints'; out.mkdir(exist_ok=True)
    base = Path(os.environ.get('CPT_BASE', str(CPT / 'checkpoints/model-ema.safetensors')))
    config = dict(run_id=RUN_ID, initialization='Studio multi-scroll CPT EMA weights (studio-multiscroll-cpt-20261023 model-ema.safetensors), fresh AdamW',
                  base_sha256=hashlib.sha256(base.read_bytes()).hexdigest(), architecture_class='DeepResidualUNet(deeper=False)', parameters=223355921,
                  total_updates=TOTAL, peak_lr=PEAK, warmup_updates=WARMUP, lr_floor_fraction=FLOOR, weight_decay=.01, gradient_clip_norm=1.,
                  effective_batch=64, microbatch=16, replay_per_update=REPLAY, replay_scrolls=list(STUDIO_SCROLLS), precision='bfloat16',
                  implementation='torch.compile(dynamic=False)', augmentation='Studio sparse_data.augment (scroll-shift-v2), unchanged',
                  loss=f'0.5 masked BCE with label smoothing {SMOOTH} (targets {SMOOTH}/{1 - SMOOTH}), per-example mean over the central 96 px averaged over '
                       'contributing examples + 0.5 microbatch Dice (hard targets, smooth 0, skipped without labeled ink) weighted by contributing examples',
                  label_smoothing=SMOOTH, dice='microbatch (v8in loss_mode=batch)', parent_run='studio-multiscroll-cpt-20261023', input_depth=16, model_window=128,
                  groups=mixture['groups'], eval_every=EVAL_EVERY, sampler_seed=2026102301, augmentation_seed=2026102302, training_seed=2026102303,
                  studio_data_sha256=hashlib.sha256((STUDIO / 'data.json').read_bytes()).hexdigest(),
                  ema_decay=EMA_DECAY, ema='EMA of all parameters, decay min(ema_decay,(1+n)/(10+n)) over the n updates since it started; evaluated and exported next to the raw weights',
                  learning_rate_formula=f'peak*min(1,(step+1)/{WARMUP})*({FLOOR}+(1-{FLOOR})*(1+cos(pi*min(step,T)/T))/2), T={TOTAL}; peak per lr_history')
    resume = torch.load(out / 'resume.pt', map_location='cpu', weights_only=False, mmap=True) if (out / 'resume.pt').exists() else None
    if resume is not None:
        for key in ('base_sha256', 'total_updates', 'warmup_updates', 'replay_per_update', 'studio_data_sha256'):
            assert resume['config'][key] == config[key], ('resume config differs', key)
    lr_history = (resume or {}).get('lr_history') or [dict(from_update=0, peak_lr=(resume or {}).get('config', config)['peak_lr'])]
    if lr_history[-1]['peak_lr'] != PEAK:
        lr_history = lr_history + [dict(from_update=resume['step'], peak_lr=PEAK)]
    config['lr_history'] = lr_history
    history = (resume or {}).get('data_history', [])
    if not history or history[-1]['rows'] != rows:
        history = history + [dict(from_update=(resume or {}).get('step', 0), rows=rows, groups=mixture['groups'], at=time.time())]
    config['data_history'] = [dict(from_update=h['from_update'], rows=len(h['rows'])) for h in history]
    atomic(ROOT / 'config.json', config)
    run = None
    if os.environ.get('STUDIO_WANDB') == '1':
        try:
            import wandb
            run = wandb.init(entity=os.environ.get('WANDB_ENTITY') or None, project=os.environ.get('WANDB_PROJECT', 'studio-multiscroll-cpt'), id=RUN_ID,
                             resume='allow', dir=str(ROOT), name='Studio · multi-scroll continued pretraining · 814/841/9B replay', config=config)
        except Exception as error:  # logging is optional; never let it stop training
            print('WANDB_DISABLED', repr(error), flush=True); run = None
    log = (ROOT / 'training.jsonl').open('a', buffering=1)

    atomic(ROOT / 'progress.json', dict(status='loading_host_atlas', updated_at=time.time()))
    atlas = HostAtlas(rows)
    groups = {name: g for name, g in mixture['groups'].items()}
    scroll_group = {s: name for name, g in groups.items() for s in g['scrolls']}
    members = defaultdict(list)
    for ri, r in enumerate(atlas.rows):
        if r['meta'].get('role', 'train') != 'train': continue
        name = scroll_group.get(r['scroll'])
        if name is None: raise ValueError(('Row scroll is in no group', r['id'], r['scroll']))
        members[name].append(np.stack([np.full(len(r['coords']), ri), np.arange(len(r['coords']))], 1))
    names = [n for n in groups if n in members]
    streams = {n: GroupStream(np.concatenate(members[n]), seed=hash_seed(n)) for n in names}
    weights = np.array([groups[n]['weight'] for n in names], float); weights /= weights.sum()
    group_rows = {n: sorted({atlas.rows[int(i)]['id'] for i in np.concatenate(members[n])[:, 0]}) for n in names}
    print('GROUPS', json.dumps({n: dict(weight=round(float(w), 4), patches=len(streams[n].entries), rows=len(group_rows[n])) for n, w in zip(names, weights)}), flush=True)
    studio_stream = StudioStream(doc); rng = np.random.default_rng(2026102301)
    if resume is not None:
        studio_stream.load(resume['studio_stream']); rng.bit_generator.state = resume['sampler_rng']
        for n in names:
            s = resume['group_streams'].get(n)
            if s and s['patches'] == len(streams[n].entries): streams[n].load(s)

    with acquire(RUN_ID, 'Studio soft-loss test (~1 h, Priyank 23 Oct 2026); checkpoints and exits cleanly on SIGTERM'):
        torch.set_num_threads(16); torch.backends.cudnn.benchmark = True; torch.backends.cuda.matmul.allow_tf32 = True; torch.set_float32_matmul_precision('high')
        torch.manual_seed(2026102303)
        model = DeepResidualUNet(deeper=False)
        if resume is not None: model.load_state_dict(resume['state_dict'])
        else:
            from safetensors.torch import load_file
            model.load_state_dict(load_file(str(base)), strict=True)
        model = model.cuda(); assert sum(p.numel() for p in model.parameters()) == 223355921
        optimizer = torch.optim.AdamW(model.parameters(), lr=PEAK, weight_decay=.01, fused=True)
        augmentation_rng = np.random.default_rng(2026102302); step = 0; prior_wall = 0.; best = None
        if resume is not None:
            optimizer.load_state_dict(resume['optimizer']); step = resume['step']; prior_wall = resume['wall_seconds']; best = resume.get('best')
            torch.set_rng_state(resume['torch_rng']); torch.cuda.set_rng_state_all(resume['cuda_rng']); augmentation_rng.bit_generator.state = resume['augmentation_rng']
        params = [p for p in model.parameters()]
        if resume is not None and resume.get('ema') is not None:
            ema = [t.to('cuda') for t in resume['ema']]; ema_updates = resume['ema_updates']; ema_from = resume['ema_from']; best_ema = resume.get('best_ema')
        else:
            ema = [p.detach().clone() for p in params]; ema_updates = 0; ema_from = step; best_ema = None
        del resume
        atomic(ROOT / 'progress.json', dict(status='loading_studio_gpu_tiles', step=step, updated_at=time.time()))
        drop_page_cache()
        data = StudioGpuData(doc); evaluator211 = Eval211(); fn = torch.compile(model, dynamic=False)
        print('GPU_FREE_AFTER_LOAD', torch.cuda.mem_get_info()[0], 'host_atlas_bytes', atlas.bytes, flush=True)
        host = REPLAY_HOST = 64 - REPLAY
        producer = cf.ThreadPoolExecutor(1); copy_stream = torch.cuda.Stream(); futures = deque(); lifetimes = deque()

        def produce():
            t = time.monotonic()
            counts = rng.multinomial(host, weights); items = []; labels = []
            for n, c in zip(names, counts):
                for ri, pi in streams[n].take(int(c)): items.append((int(ri), int(pi))); labels.append(n)
            studio_entries = studio_stream.take(REPLAY)
            offsets, params = parameters(rng, 64)
            x = torch.empty((host, 224, 224, 16), dtype=torch.uint8, pin_memory=True); y = torch.empty((host, 224, 224), dtype=torch.uint8, pin_memory=True)
            atlas.fill([(ri, pi, offsets[REPLAY + k, 0], offsets[REPLAY + k, 1]) for k, (ri, pi) in enumerate(items)], x.numpy(), y.numpy())
            groups_of = [data.rows[int(data.entry_row[e])]['scroll'] for e in studio_entries] + labels
            snapshot = dict(sampler_rng=copy.deepcopy(rng.bit_generator.state), studio_stream=studio_stream.state(),
                            group_streams={n: dict(streams[n].state(), patches=len(streams[n].entries)) for n in names})
            return dict(x=x, y=y, studio=studio_entries, offsets=offsets, params=params, groups=groups_of, snapshot=snapshot, cpu_seconds=time.monotonic() - t)

        def replenish():
            while len(futures) < 3: futures.append(producer.submit(produce))

        def upload(b):
            while lifetimes and lifetimes[0][0].query(): lifetimes.popleft()
            with torch.cuda.stream(copy_stream):
                gx = b['x'].to('cuda', non_blocking=True); gy = b['y'].to('cuda', non_blocking=True); event = torch.cuda.Event(); event.record()
            lifetimes.append((event, b)); return gx, gy, event

        consumed = None

        def checkpoint(final=False):
            payload = dict(run_id=RUN_ID, state_dict=model.state_dict(), step=step, config=config, wall_seconds=prior_wall + time.monotonic() - began)
            if not final:
                payload.update(optimizer=optimizer.state_dict(), torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all(),
                               augmentation_rng=copy.deepcopy(augmentation_rng.bit_generator.state), best=best, data_history=history, lr_history=lr_history,
                               ema=[t.detach().cpu() for t in ema], ema_updates=ema_updates, ema_from=ema_from, best_ema=best_ema,
                               **(consumed or dict(sampler_rng=copy.deepcopy(rng.bit_generator.state), studio_stream=studio_stream.state(),
                                                   group_streams={n: dict(streams[n].state(), patches=len(streams[n].entries)) for n in names})))
            path = out / ('last.pt' if final else 'resume.pt'); tmp = path.with_suffix('.tmp'); torch.save(payload, tmp); tmp.replace(path)

        def export(name):
            from safetensors.torch import save_file
            path = out / name; save_file({k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}, str(path))
            return hashlib.sha256(path.read_bytes()).hexdigest()

        def swap_ema():
            with torch.no_grad():
                for p, e in zip(params, ema):
                    t = p.detach().clone(); p.copy_(e); e.copy_(t)

        @torch.inference_mode()
        def evaluate():
            from sklearn.metrics import average_precision_score, roc_auc_score
            fn.eval(); r211 = evaluator211(fn); probabilities = []; states = []
            coords = data.validation_coords()
            for start in range(0, len(coords), 16):
                raw, state = data.validation_batch(start); n = len(raw)
                x = raw.permute(0, 3, 1, 2)[:, None].float().clamp_max_(200) / 255
                if n < 16: x = torch.cat([x, x[-1:].expand(16 - n, *x.shape[1:])])
                with torch.autocast('cuda', dtype=torch.bfloat16): z = fn(x)[:n]
                probabilities.append(z[:, 0, 48:80, 48:80].float().sigmoid().cpu().numpy()); states.append(state[:, 48:80, 48:80].cpu().numpy())
            p = np.concatenate(probabilities); s = np.concatenate(states); m = s > 0; y = s[m] == 2
            holdout = dict(ap=float(average_precision_score(y, p[m])), auc=float(roc_auc_score(y, p[m])), scope='Studio fixed 841 holdout, core 32')
            fn.train()
            return dict(pherc0211=dict(mean_ap=r211['mean_ap'], pixel_weighted_ap=r211['pixel_weighted_ap'], pooled_ap=r211['pooled_ap'],
                                       segments={x['id']: round(x['ap'], 4) for x in r211['segments']},
                                       mean_probability={x['id']: round(x['mean_probability'], 4) for x in r211['segments']}), studio_holdout=holdout)

        began = time.monotonic(); last_checkpoint = began
        if step == 0:
            scores = evaluate(); log.write(json.dumps(dict(step=0, evaluation=scores)) + '\n'); print('EVALUATION', 0, json.dumps(scores), flush=True)
            if run: run.log({'eval/pherc0211_mean_ap': scores['pherc0211']['mean_ap'], 'eval/pherc0211_pixel_weighted_ap': scores['pherc0211']['pixel_weighted_ap'], 'eval/studio_holdout_ap': scores['studio_holdout']['ap']}, step=0)
        replenish(); interval = defaultdict(float); counts_interval = defaultdict(int); t_interval = time.monotonic(); n_interval = 0; cpu_wait = 0.
        try:
            while step < TOTAL:
                if STOP:
                    checkpoint(); print('STOPPED_AT_UPDATE', step, flush=True); log.close(); sys.exit(75)
                t0 = time.monotonic(); b = futures.popleft().result(); cpu_wait += time.monotonic() - t0; replenish()
                gx, gy, event = upload(b)
                sraw, sstate = data.fetch(b['studio'], b['offsets'][:REPLAY])
                torch.cuda.current_stream().wait_event(event); gx.record_stream(torch.cuda.current_stream()); gy.record_stream(torch.cuda.current_stream())
                raw = torch.cat([sraw, gx]); state = torch.cat([sstate, gy]); del sraw, sstate, gx, gy
                x, state = augment(raw, state, b['params'], augmentation_rng); del raw
                contributing = (state[..., 16:112, 16:112].flatten(1) > 0).any(1).sum(); count = int(contributing)
                lr = learning_rate(step)
                for group in optimizer.param_groups: group['lr'] = lr
                fn.train(); optimizer.zero_grad(set_to_none=True); per_example = []
                if count:
                    for micro in range(0, 64, 16):
                        with torch.autocast('cuda', dtype=torch.bfloat16):
                            loss_sum, vector, _ = soft_terms(fn(x[micro:micro + 16]), state[micro:micro + 16]); loss = loss_sum / contributing
                        loss.backward(); per_example.append(vector)
                    norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1., error_if_nonfinite=False)
                    per_example = torch.cat(per_example).float().cpu().numpy(); active = (state[:, 0, 16:112, 16:112].flatten(1) > 0).any(1).cpu().numpy()
                    if not (np.isfinite(per_example).all() and bool(torch.isfinite(norm))):
                        # Skip a nonfinite update instead of crashing (a resume would replay the same batch).
                        optimizer.zero_grad(set_to_none=True); interval['_skipped'] += 1
                        log.write(json.dumps(dict(step=step, skipped_nonfinite=True)) + '\n'); step += 1; consumed = b['snapshot']; del x, state; continue
                    optimizer.step()
                    with torch.no_grad():
                        ema_updates += 1; torch._foreach_lerp_(ema, [p.detach() for p in params], 1 - min(EMA_DECAY, (1 + ema_updates) / (10 + ema_updates)))
                    for g, v, a in zip(b['groups'], per_example, active):
                        if a: interval[g] += float(v); counts_interval[g] += 1
                    interval['_norm'] += float(norm); interval['_loss'] += float(per_example.sum() / count)
                step += 1; n_interval += 1; consumed = b['snapshot']; del x, state
                if step <= 3 or step % 50 == 0:
                    seconds = time.monotonic() - t_interval
                    row = dict(step=step, lr=lr, loss=interval['_loss'] / n_interval, gradient_norm=interval['_norm'] / n_interval,
                               update_seconds=seconds / n_interval, cpu_wait_seconds=cpu_wait / n_interval, cpu_batch_seconds=b['cpu_seconds'],
                               group_loss={g: round(interval[g] / counts_interval[g], 4) for g in counts_interval},
                               studio_epoch=studio_stream.state(), gpu_peak_bytes=torch.cuda.max_memory_allocated(), wall_seconds=prior_wall + time.monotonic() - began)
                    log.write(json.dumps(row) + '\n'); print('UPDATE', json.dumps(row), flush=True)
                    atomic(ROOT / 'progress.json', dict(status='training', **row, total_updates=TOTAL, best=best, best_ema=best_ema, updated_at=time.time()))
                    if run: run.log({'train/' + k: v for k, v in row.items() if isinstance(v, (int, float))} | {'group_loss/' + g: v for g, v in row['group_loss'].items()}, step=step)
                    interval.clear(); counts_interval.clear(); t_interval = time.monotonic(); n_interval = 0; cpu_wait = 0.
                if step % EVAL_EVERY == 0 or step == TOTAL:
                    scores = evaluate(); sha = export(f'step-{step:06d}.safetensors'); scores['checkpoint_sha256'] = sha
                    if best is None or scores['pherc0211']['pixel_weighted_ap'] > best['pherc0211_pixel_weighted_ap']:
                        best = dict(step=step, pherc0211_pixel_weighted_ap=scores['pherc0211']['pixel_weighted_ap'], file=f'step-{step:06d}.safetensors')
                    log.write(json.dumps(dict(step=step, evaluation=scores)) + '\n'); print('EVALUATION', step, json.dumps(scores), flush=True)
                    swap_ema()
                    try:
                        escores = evaluate(); escores['checkpoint_sha256'] = export(f'ema-{step:06d}.safetensors'); escores['ema_updates'] = ema_updates; escores['ema_from'] = ema_from
                    finally:
                        swap_ema()
                    if best_ema is None or escores['pherc0211']['pixel_weighted_ap'] > best_ema['pherc0211_pixel_weighted_ap']:
                        best_ema = dict(step=step, pherc0211_pixel_weighted_ap=escores['pherc0211']['pixel_weighted_ap'], file=f'ema-{step:06d}.safetensors')
                    log.write(json.dumps(dict(step=step, evaluation_ema=escores)) + '\n'); print('EVALUATION_EMA', step, json.dumps(escores), flush=True)
                    if run: run.log({'eval/pherc0211_mean_ap': scores['pherc0211']['mean_ap'], 'eval/pherc0211_pixel_weighted_ap': scores['pherc0211']['pixel_weighted_ap'], 'eval/studio_holdout_ap': scores['studio_holdout']['ap']}, step=step)
                    checkpoint(); last_checkpoint = time.monotonic(); t_interval = time.monotonic()
                elif time.monotonic() - last_checkpoint >= 900:
                    checkpoint(); last_checkpoint = time.monotonic()
            checkpoint(); checkpoint(final=True); sha = export('model.safetensors')
            swap_ema(); ema_sha = export('model-ema.safetensors'); swap_ema()
            result = dict(status='complete', run_id=RUN_ID, global_step=step, model_sha256=sha, ema_model_sha256=ema_sha, best_pherc0211=best, best_pherc0211_ema=best_ema, config=config,
                          wall_seconds=prior_wall + time.monotonic() - began)
            atomic(ROOT / 'training-result.json', result); atomic(ROOT / 'progress.json', dict(result, updated_at=time.time()))
            print('TRAINING_COMPLETE', json.dumps(result), flush=True)
        except SystemExit:
            raise
        except BaseException as error:
            atomic(ROOT / 'progress.json', dict(status='failed', error=repr(error), step=step, updated_at=time.time())); raise
        finally:
            producer.shutdown(wait=False, cancel_futures=True)
            if run: run.finish()


def hash_seed(name):
    return int.from_bytes(hashlib.sha256(('2026102301:' + name).encode()).digest()[:4], 'little')


if __name__ == '__main__':
    def stop(signum, frame):
        global STOP
        STOP = True
    signal.signal(signal.SIGTERM, stop); signal.signal(signal.SIGINT, stop)
    main()
