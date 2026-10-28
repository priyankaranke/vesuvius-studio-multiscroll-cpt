"""Held-out PHerc0211 ink AP: Studio's published tiling (128 px window, central 96, stride 96) on the nine labeled
PHerc0211 Label Studio segments (13 Oct 2026 labels). None of these pixels is used for training.

The segments are 16-plane uint8 stacks (`ct16.npy`, [16,H,W]) with `labels.npz` (`known`, `ink`). Only stride-96 cells
whose 96 px core touches a labeled pixel are inferred. AP is computed per segment over its labeled pixels, plus a
pixel-weighted mean and a pooled AP over all labeled pixels.

  python eval211.py --checkpoint model.safetensors [--reverse] [--out result.json]
"""
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import torch

HERE = Path(__file__).resolve().parent
DATA = Path(os.environ.get('PHERC0211_EVAL_DATA', '/home/ubuntu/pherc0211-expanded9-loo-20261013/data'))


def ap(score, y):
    """Average precision (step-wise, same as the LOO analysis); y boolean."""
    order = np.argsort(-score, kind='stable'); y = y[order]; tp = np.cumsum(y)
    precision = tp / np.arange(1, len(y) + 1)
    return float((precision * y).sum() / max(y.sum(), 1))


class Eval211:
    def __init__(self, data=DATA, reverse=False):
        self.segments = []
        data = Path(data)
        if not data.exists():
            print(f'eval211: {data} not found; the PHerc0211 held-out evaluation is skipped (set PHERC0211_EVAL_DATA to enable it)', flush=True)
        for folder in sorted(data.iterdir()) if data.exists() else []:
            if not (folder / 'labels.npz').exists():
                continue
            labels = np.load(folder / 'labels.npz'); known = labels['known'].astype(bool); ink = labels['ink'].astype(bool)
            if not known.any():
                continue
            ct = np.load(folder / 'ct16.npy', mmap_mode='r')
            assert ct.shape == (16, *known.shape) and ct.dtype == np.uint8
            ct = np.ascontiguousarray(ct[::-1] if reverse else ct)
            h, w = known.shape
            cells = [(y, x) for y in range(0, h, 96) for x in range(0, w, 96) if known[y:y + 96, x:x + 96].any()]
            self.segments.append(dict(id=folder.name, ct=ct, known=known, ink=ink, cells=cells))
        self.reverse = reverse

    @staticmethod
    def _window(ct, y, x):
        h, w = ct.shape[-2:]
        tile = ct[:, max(0, y):min(h, y + 128), max(0, x):min(w, x + 128)]
        pad = ((0, 0), (max(0, -y), max(0, y + 128 - h)), (max(0, -x), max(0, x + 128 - w)))
        return np.pad(tile, pad, mode='reflect') if any(a or b for a, b in pad) else tile

    @torch.inference_mode()
    def __call__(self, model, batch=32):
        """model: callable on [B,1,16,128,128] float in [0, 200/255] -> logits [B,1,128,128]."""
        rows = []; scores = []; targets = []; began = time.monotonic()
        if not self.segments:
            nan = float('nan')
            return dict(segments=[], mean_ap=nan, pixel_weighted_ap=nan, pooled_ap=nan, reverse=self.reverse, seconds=0.0, recipe='skipped: no PHerc0211 evaluation data')
        for s in self.segments:
            h, w = s['known'].shape; prob = np.zeros((h, w), np.float32)
            for i in range(0, len(s['cells']), batch):
                cells = s['cells'][i:i + batch]
                raw = np.stack([self._window(s['ct'], y - 16, x - 16) for y, x in cells])
                x = torch.from_numpy(raw[:, None]).cuda().float().clamp_(0, 200).div_(255)
                n = len(x)
                if n < batch:
                    x = torch.cat([x, x[-1:].expand(batch - n, *x.shape[1:])])
                with torch.autocast('cuda', dtype=torch.bfloat16):
                    p = model(x)[:n, 0, 16:112, 16:112].float().sigmoid().cpu().numpy()
                for (y, xx), q in zip(cells, p):
                    hh, ww = min(96, h - y), min(96, w - xx); prob[y:y + hh, xx:xx + ww] = q[:hh, :ww]
            known = s['known']; y = s['ink'][known]; score = prob[known]
            rows.append(dict(id=s['id'], labeled=int(known.sum()), ink_fraction=float(y.mean()), ap=ap(score, y),
                             mean_probability=float(score.mean())))
            scores.append(score); targets.append(y)
        weights = np.array([r['labeled'] for r in rows], float)
        result = dict(segments=rows, mean_ap=float(np.mean([r['ap'] for r in rows])),
                      pixel_weighted_ap=float((np.array([r['ap'] for r in rows]) * weights).sum() / weights.sum()),
                      pooled_ap=ap(np.concatenate(scores), np.concatenate(targets)), reverse=self.reverse,
                      seconds=time.monotonic() - began, recipe='128 window, central 96, stride 96, bf16; labeled cells only')
        return result


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--reverse', action='store_true')
    p.add_argument('--out'); a = p.parse_args()
    sys.path.insert(0, str(HERE.parent / 'pherc1447-architecture-study-20260906'))
    from small_unets import DeepResidualUNet
    from safetensors.torch import load_file
    model = DeepResidualUNet(deeper=False); state = load_file(a.checkpoint) if a.checkpoint.endswith('.safetensors') else torch.load(a.checkpoint, map_location='cpu', weights_only=False)['state_dict']
    model.load_state_dict(state, strict=True)
    evaluator = Eval211(reverse=a.reverse)
    from gpu_lease import acquire
    with acquire('studio-multiscroll-cpt-eval211', 'PHerc0211 held-out AP of a Studio checkpoint (short evaluation)'):
        model = model.cuda().eval()
        result = dict(checkpoint=a.checkpoint, **evaluator(model))
    for r in result['segments']:
        print('%-62s labeled %8d ink %.3f AP %.4f' % (r['id'][:62], r['labeled'], r['ink_fraction'], r['ap']))
    print('mean AP %.4f  pixel-weighted %.4f  pooled %.4f  (%.0f s)' % (result['mean_ap'], result['pixel_weighted_ap'], result['pooled_ap'], result['seconds']))
    if a.out:
        Path(a.out).write_text(json.dumps(result, indent=2) + '\n')


if __name__ == '__main__':
    main()
