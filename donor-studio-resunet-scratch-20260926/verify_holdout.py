"""End-to-end check of rebuilt CT tiles: the published epoch-9 weights on the rebuilt fixed holdout, scored exactly as
train.py's evaluate() did during training (128 px holdout crops with reflect padding, clamp 200 / 255, BF16, masked
0.5 BCE + 0.5 Dice on the central 96 px, AP / ROC-AUC on the central 32 px). Eager instead of torch.compile (the run's
qualification measured a max probability difference of 0.0036 between them). The run logged at epoch 9:
loss 0.518912, AP 0.765598, AUC 0.830425 on 1,334 patches (15,791 ink / 29,558 background pixels).
Optionally also scores supervised pixels of a few training datasets (sanity: the model was trained on them).

  python verify_holdout.py [--weights hf-epoch-9/model.safetensors] [--train-datasets 6]
"""
import argparse, json, sys
from pathlib import Path
import numpy as np
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from safetensors.torch import load_file

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent / 'pherc1447-architecture-study-20260906'))
from sparse_data import masked_terms
from small_unets import DeepResidualUNet


def crops(row, coords, width=128):
    """GpuData.crops for one dataset: reflect padding into the canvas, unknown (0) outside it."""
    path = Path(row['folder']); tiles = np.load(path / 'tiles.npy'); ct = np.load(path / 'ct-tiles.npy', mmap_mode='r')
    labels = np.load(path / 'labels.npy', mmap_mode='r'); h, w = row['shape']
    lookup = {(int(y) // 512, int(x) // 512): k for k, (y, x) in enumerate(tiles)}

    def reflect(a, n):
        a = a % (2 * n - 2); return np.where(a < n, a, 2 * n - 2 - a)
    for y, x in coords:
        ys = y + np.arange(width); xs = x + np.arange(width)
        outside = (ys[:, None] < 0) | (ys[:, None] >= h) | (xs[None] < 0) | (xs[None] >= w)
        yy, xx = reflect(ys, h), reflect(xs, w)
        raw = np.empty((width, width, 16), np.uint8); state = np.empty((width, width), np.uint8)
        for ty in np.unique(yy // 512):
            for tx in np.unique(xx // 512):
                k = lookup[(int(ty), int(tx))]; my = (yy // 512) == ty; mx = (xx // 512) == tx
                raw[np.ix_(my, mx)] = ct[k][np.ix_(yy[my] % 512, xx[mx] % 512)]
        state[:] = labels[np.ix_(yy, xx)]; state[outside] = 0
        yield raw, state


def score(model, row, coords):
    loss_sum = 0.; examples = 0; probs = []; states = []; batch = []
    def flush():
        nonlocal loss_sum, examples
        raw = torch.from_numpy(np.stack([b[0] for b in batch])).cuda(); state = torch.from_numpy(np.stack([b[1] for b in batch])).cuda(); n = len(raw)
        x = raw.permute(0, 3, 1, 2)[:, None].float().clamp_max_(200) / 255
        if n < 16: x = torch.cat([x, x[-1:].expand(16 - n, *x.shape[1:])])
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            z = model(x)[:n]; total, count = masked_terms(z, state[:, None])
        loss_sum += float(total); examples += int(count)
        probs.append(z[:, 0, 48:80, 48:80].float().sigmoid().cpu().numpy()); states.append(state[:, 48:80, 48:80].cpu().numpy()); batch.clear()
    for item in crops(row, coords):
        batch.append(item)
        if len(batch) == 16: flush()
    if batch: flush()
    p = np.concatenate(probs); s = np.concatenate(states); m = s > 0; y = s[m] == 2; q = p[m]
    both = bool(len(y) and y.any() and (~y).any())
    return dict(loss=loss_sum / examples if examples else None, ap=float(average_precision_score(y, q)) if both else None,
                auc=float(roc_auc_score(y, q)) if both else None, patches=len(p), supervised_ink=int(y.sum()), supervised_background=int((~y).sum()))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--weights', default=str(ROOT / 'hf-epoch-9/model.safetensors'))
    ap.add_argument('--train-datasets', type=int, default=6); a = ap.parse_args()
    doc = json.loads((ROOT / 'data.json').read_text()); rows = [r for r in doc['segments'] if r['samples']]
    model = DeepResidualUNet(deeper=False); model.load_state_dict(load_file(a.weights), strict=True); model.cuda().eval()
    hold = next(r for r in rows if r['role'] == 'validation')
    result = dict(holdout=score(model, hold, np.load(Path(hold['folder']) / 'coordinates.npy')),
                  logged_epoch9=dict(loss=0.518912, ap=0.765598, auc=0.830425, patches=1334, supervised_ink=15791, supervised_background=29558))
    print('HOLDOUT', json.dumps(result['holdout']), flush=True)
    rng = np.random.default_rng(0); train = [r for r in rows if r['role'] == 'train' and (Path(r['folder']) / 'render.json').exists() and json.loads((Path(r['folder']) / 'render.json').read_text()).get('status') == 'complete']
    for r in rng.choice(train, size=min(a.train_datasets, len(train)), replace=False):
        c = np.load(Path(r['folder']) / 'coordinates.npy'); c = c[rng.choice(len(c), size=min(256, len(c)), replace=False)]
        result[r["id"]] = score(model, r, c - 64); print('TRAIN', r['id'], r['geometry']['kind'], json.dumps(result[r['id']]), flush=True)
    (ROOT / 'rebuild/verify-holdout.json').write_text(json.dumps(result, indent=1))


if __name__ == '__main__':
    main()
