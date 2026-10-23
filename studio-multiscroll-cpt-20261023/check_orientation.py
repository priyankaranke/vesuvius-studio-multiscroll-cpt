"""Depth-order check of prepared rows against Studio: AP of the stock Studio checkpoint on sampled training patches of
each row, forward and depth-reversed. A row whose forward AP is not clearly above its reversed AP is reported.

  python check_orientation.py --checkpoint model.safetensors --rows DIR [DIR ...] [--per-row 96] --out result.json
"""
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'pherc1447-architecture-study-20260906'))
from multiscroll_data import HostAtlas
from eval211 import ap


def main():
    p = argparse.ArgumentParser(); p.add_argument('--checkpoint', required=True); p.add_argument('--rows', nargs='+', required=True)
    p.add_argument('--per-row', type=int, default=96); p.add_argument('--out', required=True); a = p.parse_args()
    from small_unets import DeepResidualUNet
    from safetensors.torch import load_file
    from gpu_lease import acquire
    atlas = HostAtlas(a.rows); rng = np.random.default_rng(0); results = []
    model = DeepResidualUNet(deeper=False); model.load_state_dict(load_file(a.checkpoint), strict=True)
    with acquire('studio-multiscroll-cpt-orientation', 'Depth-order check of prepared training rows (short)'), torch.inference_mode():
        model = model.cuda().eval()
        for ri, r in enumerate(atlas.rows):
            pick = rng.choice(len(r['coords']), min(a.per_row, len(r['coords'])), replace=False)
            crops = [atlas.crop(ri, int(r['coords'][i][0]), int(r['coords'][i][1]), 128) for i in pick]
            raw = torch.from_numpy(np.stack([c[0] for c in crops])).permute(0, 3, 1, 2)[:, None].float().clamp_(0, 200).div_(255)
            state = np.stack([c[1] for c in crops])[:, 16:112, 16:112]
            row = dict(id=r['id'], scroll=r['scroll'], patches=len(pick))
            for name, flip in (('forward', False), ('reverse', True)):
                probs = []
                for i in range(0, len(raw), 32):
                    x = raw[i:i + 32].cuda(); x = x.flip(2) if flip else x
                    with torch.autocast('cuda', dtype=torch.bfloat16):
                        probs.append(model(x)[:, 0, 16:112, 16:112].float().sigmoid().cpu().numpy())
                prob = np.concatenate(probs); known = state > 0; y = state[known] == 2
                row[name] = ap(prob[known], y) if y.any() and (~y).any() else None
                row['ink_fraction'] = float(y.mean()) if len(y) else None
            row['forward_better'] = None if row['forward'] is None else bool(row['forward'] > row['reverse'])
            results.append(row); print(json.dumps(row), flush=True)
    Path(a.out).write_text(json.dumps(results, indent=2) + '\n')


if __name__ == '__main__':
    main()
