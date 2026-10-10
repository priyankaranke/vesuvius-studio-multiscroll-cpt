"""Re-render the Studio run's native CT tiles (prepared/<id>/ct-tiles.npy) after the run's local scan copies were deleted.

The original run rendered every tile from complete local copies of the three public scans (render.py; data.json
storage: "Reuse complete native acquisition caches"). Those copies are gone. This wrapper restores exactly the chunks
the training tiles read and then calls the unchanged render.py functions:

  plan    campaign tiles: the renderer's own conservative chunk plan for each tile (render.py's campaign_tile command
          with --chunk-plan-output), union per dataset; written to rebuild/plans/<id>.json. Bytes from the backup's S3
          object inventories (donor-spirals-20260921/inputs/<scroll>-ct.objects.jsonl).
  run     per dataset: download the planned chunks from the public bucket into the scan's local zarr
          (scan.json `destination`; each chunk checked against its inventory size and ETag/MD5; chunks absent from the
          inventory are outside the published mask, and absent locally as in the original copy), then render.render_row.
          Original-donor rows use render.py's accepted sampler; their local chunk reader downloads a missing chunk (same
          checks) before reading it. Chunks no pending dataset needs are deleted after each dataset (disk reserve).
  verify  every eligible dataset has render.json complete and ct-tiles.npy of shape [tiles, 512, 512, 16] uint8.

data.json is never rewritten. Downloads are rate-limited (STUDIO_CT_RATE_MIB, default 30 MiB/s per process group).

  python render_rebuild.py plan [--workers 24]
  python render_rebuild.py run [--render-workers 12] [--min-free-gib 230]
  python render_rebuild.py verify
"""
import argparse, collections, hashlib, json, os, shutil, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from pathlib import Path
import numpy as np
import requests
import render
from prepare import ROOT, STUDIO, read, atomic

OUT = ROOT / 'rebuild'; PLANS = OUT / 'plans'
SCROLL_DIR = {'PHerc0814': '814', 'PHerc0841': '841', 'PHerc0009B': '9B'}
RATE = float(os.environ.get('STUDIO_CT_RATE_MIB', '30')) * 2 ** 20


def rows():
    return [r for r in read(ROOT / 'data.json')['segments'] if r['samples']]


# ---------------------------------------------------------------- inventory and verified download
_inv = {}


def inventory(scroll):
    """{'z/y/x': (bytes, etag)} for level 0 of the scroll's scan, from the S3 object listing in the backup."""
    s = SCROLL_DIR[scroll]
    if s not in _inv:
        d = {}
        for line in (STUDIO / 'inputs' / f'{s}-ct.objects.jsonl').open():
            o = json.loads(line); k = o['key']; i = k.find('.zarr/0/')
            if i >= 0: d[k[i + 8:]] = (int(o['size']), o['etag'].strip('"'))
        _inv[s] = d
    return _inv[s]


class Limiter:
    def __init__(self, rate): self.rate, self.t, self.lock = rate, time.monotonic(), threading.Lock()

    def take(self, n):
        with self.lock:
            self.t = max(self.t, time.monotonic()) + n / self.rate; wait = self.t - time.monotonic()
        if wait > 0: time.sleep(wait)


LIMIT = Limiter(RATE); _session = threading.local()


_scans = {}


def _scan(scroll):
    if scroll not in _scans: _scans[scroll] = read(STUDIO / SCROLL_DIR[scroll] / 'config/scan.json')
    return _scans[scroll]


def download(scroll, key):
    """Fetch level-0 chunk `z/y/x` into the local zarr; False when the chunk is outside the published mask."""
    scan = _scan(scroll); path = Path(scan['destination']) / '0' / key
    if path.exists(): return True
    meta = inventory(scroll).get(key)
    if meta is None: return False
    size, etag = meta
    if not hasattr(_session, 's'): _session.s = requests.Session()
    for attempt in range(6):
        try:
            LIMIT.take(size)
            r = _session.s.get(scan['source_url'] + '0/' + key, timeout=120); r.raise_for_status(); b = r.content
            if len(b) != size or ('-' not in etag and hashlib.md5(b).hexdigest() != etag): raise IOError(f'checksum {key}')
            path.parent.mkdir(parents=True, exist_ok=True); tmp = path.with_suffix('.part'); tmp.write_bytes(b); tmp.replace(path)
            return True
        except (requests.RequestException, IOError):
            time.sleep(2 ** attempt)
    raise RuntimeError(f'download failed {scroll} {key}')


# ---------------------------------------------------------------- campaign plans (render.campaign_tile's command)
def plan_tile(row, y, x, scratch):
    g = row['geometry']; oy, ox = g['offset_yx']; h, w = row['shape']; hh, ww = min(512, h - y), min(512, w - x)
    scratch.mkdir(parents=True, exist_ok=True)
    cmd = ['sudo', '-n', 'docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}',
           '--cpus', '1', '--memory', '4g', '--memory-swap', '4g', '--label', 'vesuvius.task=studio-training-rebuild-20261010',
           '-e', 'OMP_NUM_THREADS=1', '-e', 'CUDA_VISIBLE_DEVICES=',
           '-v', f'{STUDIO}:/production:ro', '-v', f'{ROOT}:/training', '-v', f'{render.RUNTIME}:/runtime:ro',
           '--entrypoint', '/runtime/vc_render_tifxyz', render.IMAGE,
           '-v', '/production/' + str(Path(g['scan']['destination']).relative_to(STUDIO)),
           '-g', '0', '--scale', '1', '-s', '/production/' + str(Path(g['mesh']).relative_to(STUDIO)),
           '-n', '17', '--slice-step', '1', '--pyramid', '0',
           '--tif-output', '/training/' + str((scratch / 'plan-only').relative_to(ROOT)),
           '--chunk-plan-output', '/training/' + str((scratch / 'plan.json').relative_to(ROOT)),
           '--voxel-size', str(g['pitch']), '--voxel-unit', 'micrometer',
           '--crop-x', str(int(x + ox)), '--crop-y', str(int(y + oy)), '--crop-width', str(ww), '--crop-height', str(hh)]
    with (scratch / 'plan.log').open('w') as log: subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
    p = read(scratch / 'plan.json'); shutil.rmtree(scratch)
    return ['/'.join(map(str, c)) for c in p['chunks']]


def plan(workers):
    PLANS.mkdir(parents=True, exist_ok=True); todo = []
    for row in rows():
        if row['geometry']['kind'] != 'campaign' or (PLANS / f"{row['id']}.json").exists(): continue
        todo.append(row)
    print('PLAN datasets', len(todo), flush=True)
    with ThreadPoolExecutor(workers) as pool:
        for row in todo:
            tiles = np.load(Path(row['folder']) / 'tiles.npy')
            futs = [pool.submit(plan_tile, row, int(y), int(x), OUT / 'scratch' / f"plan-{row['id']}-{i}") for i, (y, x) in enumerate(tiles)]
            keys = set()
            for f in futs: keys.update(f.result())
            inv = inventory(row['scroll']); present = [k for k in keys if k in inv]
            atomic(PLANS / f"{row['id']}.json", dict(id=row['id'], scroll=row['scroll'], tiles=len(tiles), keys=sorted(keys),
                                                    in_mask=len(present), bytes=sum(inv[k][0] for k in present)))
            print('PLANNED', row['id'], len(tiles), 'tiles', len(keys), 'chunks', len(present), 'in mask', flush=True)
    summarize()


def summarize():
    per = collections.defaultdict(set); tiles = collections.Counter()
    for f in PLANS.glob('*.json'):
        p = read(f); inv = inventory(p['scroll']); per[p['scroll']].update(k for k in p['keys'] if k in inv); tiles[p['scroll']] += p['tiles']
    total = 0
    for s, ks in per.items():
        b = sum(inventory(s)[k][0] for k in ks); total += b
        print('SCROLL', s, 'tiles', tiles[s], 'unique chunks', len(ks), f'{b / 2 ** 30:.1f} GiB', flush=True)
    print('TOTAL campaign download', f'{total / 2 ** 30:.1f} GiB', f'~{total / RATE / 3600:.1f} h at {RATE / 2 ** 20:.0f} MiB/s', flush=True)


# ---------------------------------------------------------------- run
_patched = False


def patch_original_sampler(rate):
    """render.original_source reads the scan's local copy through FastVolume; a missing chunk there would read as zeros.
    Make FastVolume download a missing in-mask chunk (verified) into the local copy before its unchanged loader reads it."""
    global _patched
    if _patched: return
    import fast_volume_sampling
    LIMIT.rate = rate; dests = {}
    for scroll, s in SCROLL_DIR.items():
        dests[str(Path(_scan(scroll)['destination']) / '0') + '/'] = scroll
    load = fast_volume_sampling.FastVolume._load

    def _load(self, key):
        scroll = dests.get(self.url)
        if scroll is not None:
            sep = self.meta.get('dimension_separator', '.'); download(scroll, sep.join(map(str, key)))
        return load(self, key)
    fast_volume_sampling.FastVolume._load = _load; _patched = True


def render_one(row_id, rate):
    row = next(r for r in rows() if r['id'] == row_id)
    if row['geometry']['kind'] == 'original': patch_original_sampler(rate)
    return render.render_row(row)


def run(render_workers, min_free_gib):
    data = rows(); need = collections.Counter(); plans = {}
    for r in data:
        if r['geometry']['kind'] == 'campaign':
            p = read(PLANS / f"{r['id']}.json"); plans[r['id']] = p; need.update((r['scroll'], k) for k in p['keys'])
    done = {r['id'] for r in data if (Path(r['folder']) / 'render.json').exists() and read(Path(r['folder']) / 'render.json').get('status') == 'complete'}
    order = [r for r in sorted(data, key=lambda r: (r['geometry']['kind'] != 'campaign', r['scroll'], r['tiles'])) if r['id'] not in done]
    print('RUN datasets', len(order), 'already complete', len(done), flush=True)

    def release(r):
        if r['id'] not in plans: return
        dest = Path(_scan(r['scroll'])['destination']) / '0'
        for k in plans[r['id']]['keys']:
            need[(r['scroll'], k)] -= 1
            if need[(r['scroll'], k)] <= 0: (dest / k).unlink(missing_ok=True)

    with ThreadPoolExecutor(8) as stagers, ProcessPoolExecutor(render_workers) as renderers:
        staged = {}; rendering = {}
        for r in order:
            while shutil.disk_usage(ROOT).free / 2 ** 30 < min_free_gib and rendering:
                fin = next(as_completed(list(rendering.values()))); rid = next(k for k, v in rendering.items() if v is fin)
                fin.result(); release(next(x for x in data if x['id'] == rid)); rendering.pop(rid); print('READY', rid, flush=True)
            if r['id'] in plans:
                list(stagers.map(lambda k: download(r['scroll'], k), plans[r['id']]['keys']))
            rendering[r['id']] = renderers.submit(render_one, r['id'], RATE / render_workers); print('STAGED', r['id'], flush=True)
            for rid, f in list(rendering.items()):
                if f.done(): f.result(); release(next(x for x in data if x['id'] == rid)); rendering.pop(rid); print('READY', rid, flush=True)
        for rid, f in rendering.items():
            f.result(); release(next(x for x in data if x['id'] == rid)); print('READY', rid, flush=True)
    verify()


def verify():
    bad = 0; total = 0
    for r in rows():
        f = Path(r['folder']); ok = (f / 'render.json').exists() and read(f / 'render.json').get('status') == 'complete'
        if ok:
            a = np.load(f / 'ct-tiles.npy', mmap_mode='r'); ok = a.shape == (r['tiles'], 512, 512, 16) and a.dtype == np.uint8; total += a.nbytes
        bad += not ok
        if not ok: print('INCOMPLETE', r['id'], flush=True)
    print('VERIFY datasets', len(rows()), 'incomplete', bad, f'ct-tiles {total / 2 ** 30:.1f} GiB', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('action', choices=['plan', 'run', 'verify', 'summary'])
    ap.add_argument('--workers', type=int, default=24); ap.add_argument('--render-workers', type=int, default=12)
    ap.add_argument('--min-free-gib', type=float, default=230); a = ap.parse_args()
    {'plan': lambda: plan(a.workers), 'run': lambda: run(a.render_workers, a.min_free_gib), 'verify': verify, 'summary': summarize}[a.action]()
