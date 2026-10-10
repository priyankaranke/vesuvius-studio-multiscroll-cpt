"""Compare a training run's log with the original run's, update by update.

The data order, sampler, augmentation draws and learning-rate schedule are deterministic, so the learning rate,
contributing-example and empty-example counts must match exactly. Loss and gradient norm come from BF16 GPU arithmetic
with torch.compile, so they match closely but not bit-for-bit across machines and library builds.

  python compare_runs.py donor-studio-resunet-scratch-20260926/training.jsonl [--steps 300]
"""
import argparse, json
from pathlib import Path
import numpy as np

REF = Path(__file__).resolve().parent / 'donor-studio-resunet-scratch-20260926/reference/training-20260926.jsonl'


def updates(path):
    out = {}
    for line in Path(path).read_text().splitlines():
        r = json.loads(line)
        if 'loss' in r: out[r['step']] = r
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('log'); ap.add_argument('--steps', type=int, default=0); a = ap.parse_args()
    ref, new = updates(REF), updates(a.log)
    steps = sorted(s for s in new if s in ref and (not a.steps or s <= a.steps))
    if not steps: raise SystemExit('no overlapping updates')
    exact = {k: all(ref[s][k] == new[s][k] for s in steps) for k in ('lr', 'contributing_examples', 'empty_examples', 'offset', 'epoch')}
    rl, nl = np.array([ref[s]['loss'] for s in steps]), np.array([new[s]['loss'] for s in steps])
    rg, ng = np.array([ref[s]['gradient_norm'] for s in steps]), np.array([new[s]['gradient_norm'] for s in steps])
    rel = np.abs(nl - rl) / np.abs(rl)
    print(f'updates compared: {len(steps)} (steps {steps[0]}-{steps[-1]})')
    for k, v in exact.items(): print(f'  {k:22s} identical: {v}')
    print(f'  loss         first {nl[0]:.6f} vs {rl[0]:.6f}; mean {nl.mean():.6f} vs {rl.mean():.6f}; '
          f'|rel diff| median {np.median(rel):.2e}, max {rel.max():.2e}')
    print(f'  gradient norm mean {ng.mean():.4f} vs {rg.mean():.4f}; corr of loss curves {np.corrcoef(nl, rl)[0, 1]:.4f}')
    for s in steps[:5]: print(f'    step {s}: loss {new[s]["loss"]:.6f} vs {ref[s]["loss"]:.6f}')
    good = all(exact.values()) and abs(nl.mean() - rl.mean()) / rl.mean() < 0.01
    print('REPRODUCED' if good else 'DIFFERS', flush=True)


if __name__ == '__main__':
    main()
