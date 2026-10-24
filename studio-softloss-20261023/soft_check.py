"""Step 1 of the soft-prediction question (2026-10-23): is the softness of Youssef v8in already in Studio's ranking?

AP is unchanged by any monotone remapping of a model's output, so if Studio · multi-scroll ranks the held-out PHerc0211
labels at least as well as v8in, the visible difference is mostly the value scale (fixable by display, e.g. log-odds),
not missing evidence.

  infer    full-segment logits of a Studio-architecture checkpoint on the nine held-out PHerc0211 Label Studio
           segments (eval211's recipe: 128 px window, central 96, stride 96, bf16), every cell touching the support mask
  analyze  per-segment AP and output distributions on labeled pixels inside v8in's coverage (the v8in maps are the
           Oct 13 Youssef LOO base predictions of YoussefMoNader/ink-8um-v8in, official 64 px / stride 21 recipe), and
           review panels: CT | labels | v8in | Studio prob | Studio log-odds, display stretch 1-99.5 % of supported pixels

  python soft_check.py infer --checkpoint X.safetensors --name studioms
  python soft_check.py analyze --models studioms[,softloss-ema-003000]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('PHERC0211_EVAL_DATA', '/home/ubuntu/pherc0211-expanded9-loo-20261013/data'))
YOUSSEF = Path(os.environ.get('YOUSSEF_LOO', '/home/ubuntu/pherc0211-youssef-loo-20261013'))
OUT = HERE / 'soft-check'


def ap(score, y):
    order = np.argsort(-score, kind='stable'); y = y[order]; tp = np.cumsum(y)
    return float(((tp / np.arange(1, len(y) + 1)) * y).sum() / max(y.sum(), 1))


def segments():
    for folder in sorted(DATA.iterdir()):
        if not (folder / 'labels.npz').exists(): continue
        labels = np.load(folder / 'labels.npz'); known = labels['known'].astype(bool)
        if known.any(): yield folder, known, labels['ink'].astype(bool)


def window(ct, y, x):
    h, w = ct.shape[-2:]
    tile = ct[:, max(0, y):min(h, y + 128), max(0, x):min(w, x + 128)]
    pad = ((0, 0), (max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w)))
    return np.pad(tile, pad, mode='reflect') if any(a or b for a, b in pad) else tile


def infer(a):
    import torch
    sys.path.insert(0, str(HERE.parent / 'pherc1447-architecture-study-20260906'))
    from small_unets import DeepResidualUNet
    from safetensors.torch import load_file
    model = DeepResidualUNet(deeper=False); model.load_state_dict(load_file(a.checkpoint), strict=True); model = model.cuda().eval()
    out = OUT / 'maps' / a.name; out.mkdir(parents=True, exist_ok=True); began = time.monotonic()
    with torch.inference_mode():
        for folder, known, _ in segments():
            ct = np.ascontiguousarray(np.load(folder / 'ct16.npy', mmap_mode='r')); support = np.load(folder / 'support.npy').astype(bool) | known
            h, w = known.shape; s = a.stride; start = 0 if s == 96 else -s
            # stride 96: eval211's tiling (each pixel from one window). Smaller strides: Hann-weighted mean of the overlapping
            # 96 px cores' logits (display check only; AP comparisons use stride 96 for every model).
            hann = np.hanning(98)[1:-1].astype(np.float32); weight = np.outer(hann, hann) + 1e-3
            total = np.zeros((h, w), np.float32); norm = np.zeros((h, w), np.float32)
            cells = [(y, x) for y in range(start, h, s) for x in range(start, w, s) if support[max(0, y):y + 96, max(0, x):x + 96].any()]
            for i in range(0, len(cells), 32):
                batch = cells[i:i + 32]
                x = torch.from_numpy(np.stack([window(ct, y - 16, xx - 16) for y, xx in batch])[:, None]).cuda().float().clamp_(0, 200).div_(255)
                n = len(x)
                if n < 32: x = torch.cat([x, x[-1:].expand(32 - n, *x.shape[1:])])
                with torch.autocast('cuda', dtype=torch.bfloat16): z = model(x)[:n, 0, 16:112, 16:112].float().cpu().numpy()
                for (y, xx), q in zip(batch, z):
                    y0, x0 = max(0, y), max(0, xx); y1, x1 = min(h, y + 96), min(w, xx + 96)
                    qq = q[y0 - y:y1 - y, x0 - xx:x1 - xx]; ww = weight[y0 - y:y1 - y, x0 - xx:x1 - xx] if s != 96 else 1.
                    total[y0:y1, x0:x1] += qq * ww; norm[y0:y1, x0:x1] += ww
            with np.errstate(invalid='ignore', divide='ignore'): logit = np.where(norm > 0, total / norm, np.nan).astype(np.float32)
            np.save(out / f'{folder.name}.npy', logit); print('INFERRED', folder.name, len(cells), 'cells', flush=True)
    (out / 'meta.json').write_text(json.dumps(dict(checkpoint=a.checkpoint, recipe=f'128 window, central 96, stride {a.stride}'
                                                   + (' (Hann-weighted logit mean)' if a.stride != 96 else '') + ', bf16, all support cells',
                                                   seconds=time.monotonic() - began), indent=2) + '\n')


def v8in_maps():
    folds = json.loads((YOUSSEF / 'folds.json').read_text())['folds']
    return {f['id']: Path(f['path']) / 'inference' for f in folds}


def stats(p, known, ink):
    bg = p[known & ~ink]; fg = p[known & ink]
    q = lambda v: [round(float(t), 4) for t in np.quantile(v, [.1, .5, .9])] if len(v) else None
    return dict(background_q10_50_90=q(bg), ink_q10_50_90=q(fg), labeled_mean=round(float(p[known].mean()), 4))


def stretch(v, lo, hi):
    return np.clip((v - lo) / max(hi - lo, 1e-9), 0, 1)


def analyze(a):
    import matplotlib; matplotlib.use('Agg')
    from PIL import Image
    turbo = matplotlib.colormaps['turbo']
    models = a.models.split(','); youssef = v8in_maps(); rows = []; panels = []
    for folder, known, ink in segments():
        base = youssef[folder.name]
        v8 = np.load(base / 'base.npy'); cover = np.load(base / 'base-coverage.npy').astype(bool)
        common = known & cover
        for name in models:
            common &= np.isfinite(np.load(OUT / 'maps' / name / f'{folder.name}.npy', mmap_mode='r'))
        y = ink[common]; row = dict(id=folder.name, labeled=int(known.sum()), compared=int(common.sum()), ink_fraction=round(float(y.mean()), 4),
                                    v8in=dict(ap=round(ap(v8[common], y), 4), **stats(v8, common, ink)))
        logits = {}
        for name in models:
            z = np.load(OUT / 'maps' / name / f'{folder.name}.npy'); logits[name] = z; p = 1 / (1 + np.exp(-np.nan_to_num(z, nan=-30)))
            row[name] = dict(ap=round(ap(z[common], y), 4), **stats(p, common, ink),
                             logit_background_q10_50_90=[round(float(t), 2) for t in np.quantile(z[common & ~ink], [.1, .5, .9])],
                             logit_ink_q10_50_90=[round(float(t), 2) for t in np.quantile(z[common & ink], [.1, .5, .9])])
        rows.append(row); print(json.dumps(row), flush=True)
        if folder.name in a.panels.split(','):
            panels.append(panel(folder, known, ink, cover, v8, logits, models, turbo))
    weights = np.array([r['compared'] for r in rows], float)
    summary = {k: round(float(sum(r[k]['ap'] * wt for r, wt in zip(rows, weights)) / weights.sum()), 4) for k in ['v8in'] + models}
    result = dict(segments=rows, pixel_weighted_ap=summary, scope='labeled pixels inside v8in coverage and every compared map; 13 Oct labels')
    (OUT / f'result{a.tag}.json').write_text(json.dumps(result, indent=2) + '\n'); print('PIXEL_WEIGHTED_AP', json.dumps(summary), flush=True)
    for i, image in enumerate(panels):
        Image.fromarray(image).save(OUT / f'panel{a.tag}-{i + 1}.jpg', quality=90)


def panel(folder, known, ink, cover, v8, logits, models, turbo, side=1400, out=560):
    ys, xs = np.nonzero(known); cy, cx = int(np.median(ys)), int(np.median(xs)); h, w = known.shape
    y0 = int(np.clip(cy - side // 2, 0, max(0, h - side))); x0 = int(np.clip(cx - side // 2, 0, max(0, w - side)))
    sl = (slice(y0, y0 + side), slice(x0, x0 + side))
    ct = np.load(folder / 'ct16.npy', mmap_mode='r')[8][sl].astype(np.float32)
    supported = np.isfinite(logits[models[0]]) & cover

    def disp(v, valid, log=False):
        vals = v[valid]; lo, hi = np.quantile(vals, [.01, .995]); img = turbo(stretch(v[sl], lo, hi))[..., :3]
        img[~valid[sl]] = .15; return img
    gray = np.repeat(stretch(ct, *np.quantile(ct, [.01, .99]))[..., None], 3, 2)
    lab = gray.copy(); lab[known[sl] & ink[sl]] = (1, .2, .2); lab[known[sl] & ~ink[sl]] = lab[known[sl] & ~ink[sl]] * .6 + np.array([.2, .4, 1]) * .4
    tiles = [gray, lab, disp(v8, cover)]
    titles = ['CT (plane 8)', 'labels (red ink / blue bg)', 'v8in prob, 1-99.5%']
    for name in models:
        z = logits[name]; p = 1 / (1 + np.exp(-np.nan_to_num(z, nan=-30)))
        tiles += [disp(p, supported), disp(np.nan_to_num(z, nan=-30), supported)]
        titles += [f'{name} prob, 1-99.5%', f'{name} log-odds, 1-99.5%']
    from PIL import Image, ImageDraw
    row = []
    for t, title in zip(tiles, titles):
        im = Image.fromarray((t * 255).astype(np.uint8)).resize((out, out), Image.BILINEAR); d = ImageDraw.Draw(im)
        d.rectangle([0, 0, out, 18], fill=(0, 0, 0)); d.text((4, 3), title, fill=(255, 255, 255)); row.append(np.asarray(im))
    strip = np.concatenate(row, 1); head = Image.new('RGB', (strip.shape[1], 22), (0, 0, 0))
    ImageDraw.Draw(head).text((4, 5), f'{folder.name}  crop y{y0} x{x0} {side}px', fill=(255, 255, 255))
    return np.concatenate([np.asarray(head), strip], 0)


if __name__ == '__main__':
    p = argparse.ArgumentParser(); sub = p.add_subparsers(dest='cmd', required=True)
    i = sub.add_parser('infer'); i.add_argument('--checkpoint', required=True); i.add_argument('--name', required=True)
    i.add_argument('--stride', type=int, default=96)
    n = sub.add_parser('analyze'); n.add_argument('--models', required=True); n.add_argument('--tag', default='')
    n.add_argument('--panels', default='PHerc0211-P737m01st3-grown,PHerc0211-3305-r002-c01-manual-20261008-203736,PHerc0211-r005-c01-manual-20261008-195107')
    a = p.parse_args(); infer(a) if a.cmd == 'infer' else analyze(a)
