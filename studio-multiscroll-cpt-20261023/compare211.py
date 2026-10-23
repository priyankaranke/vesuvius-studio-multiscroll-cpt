"""Side-by-side review sheet on labeled PHerc0211 segments: CT (mean of the 16 planes), labels (ink white, background gray,
unknown black), stock Studio and a new checkpoint (probability, 0-1 gray), each with the per-segment AP.
Uses eval211's tiling (labeled stride-96 cells only; other cells stay black).

  python compare211.py --base studio.safetensors --new model.safetensors --out sheet.png [--segments ID ...] [--crop 1400]
"""
import argparse, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image, ImageDraw
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pherc1447-architecture-study-20260906'))
from eval211 import Eval211, ap


def predict(model, s, batch=32):
    h, w = s['known'].shape; prob = np.zeros((h, w), np.float32)
    with torch.inference_mode():
        for i in range(0, len(s['cells']), batch):
            cells = s['cells'][i:i + batch]
            x = torch.from_numpy(np.stack([Eval211._window(s['ct'], y - 16, x - 16) for y, x in cells])[:, None]).cuda().float().clamp_(0, 200).div_(255)
            with torch.autocast('cuda', dtype=torch.bfloat16):
                p = model(x)[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
            for (y, xx), q in zip(cells, p):
                hh, ww = min(96, h - y), min(96, w - xx); prob[y:y + hh, xx:xx + ww] = q[:hh, :ww]
    return prob


def main():
    p = argparse.ArgumentParser(); p.add_argument('--base', required=True); p.add_argument('--new', required=True); p.add_argument('--out', required=True)
    p.add_argument('--segments', nargs='*'); p.add_argument('--crop', type=int, default=1400); p.add_argument('--new-label', default='multi-scroll CPT'); a = p.parse_args()
    from small_unets import DeepResidualUNet
    from safetensors.torch import load_file
    from gpu_lease import acquire
    ev = Eval211(); segs = [s for s in ev.segments if not a.segments or s['id'] in a.segments]
    models = {}
    for name, path in (('Studio (stock)', a.base), (a.new_label, a.new)):
        m = DeepResidualUNet(deeper=False); m.load_state_dict(load_file(path), strict=True); models[name] = m
    rows = []
    with acquire('studio-multiscroll-cpt-compare211', 'Review sheet: stock vs new Studio on labeled PHerc0211 (short)'):
        preds = {}
        for name, m in models.items():
            m = m.cuda().eval(); preds[name] = [predict(m, s) for s in segs]; m.cpu()
    for i, s in enumerate(segs):
        known = s['known']; c = a.crop; h, w = known.shape
        # crop window with the most labeled pixels
        dens = np.add.reduceat(np.add.reduceat(known.astype(np.int32), np.arange(0, h, 64), 0), np.arange(0, w, 64), 1)
        k = min(c // 64, dens.shape[0]), min(c // 64, dens.shape[1])
        best = (0, 0, -1)
        for y in range(0, dens.shape[0] - k[0] + 1):
            for x in range(0, dens.shape[1] - k[1] + 1):
                v = dens[y:y + k[0], x:x + k[1]].sum()
                if v > best[2]: best = (y * 64, x * 64, v)
        y0, x0 = best[0], best[1]; sl = (slice(y0, y0 + c), slice(x0, x0 + c))
        ct = np.clip(s['ct'][:, sl[0], sl[1]].mean(0) / 200 * 255, 0, 255).astype(np.uint8)
        lab = np.where(known[sl], np.where(s['ink'][sl], 255, 90), 0).astype(np.uint8)
        panels = [('CT', ct), ('labels', lab)]
        for name in models:
            pr = preds[name][i]; score = ap(pr[known], s['ink'][known])
            panels.append((f'{name}  AP {score:.3f}', (np.clip(pr[sl], 0, 1) * 255).astype(np.uint8)))
        tile = Image.new('L', (len(panels) * ct.shape[1], ct.shape[0] + 28), 0); d = ImageDraw.Draw(tile)
        for j, (t, im) in enumerate(panels):
            tile.paste(Image.fromarray(im), (j * ct.shape[1], 28)); d.text((j * ct.shape[1] + 6, 6), t, fill=255)
        d.text((6 + 2 * ct.shape[1], 16), s['id'][:70], fill=200); rows.append(tile)
    sheet = Image.new('L', (max(r.width for r in rows), sum(r.height for r in rows)), 0); y = 0
    for r in rows: sheet.paste(r, (0, y)); y += r.height
    sheet.save(a.out); print('SHEET', a.out, sheet.size)


if __name__ == '__main__':
    main()
