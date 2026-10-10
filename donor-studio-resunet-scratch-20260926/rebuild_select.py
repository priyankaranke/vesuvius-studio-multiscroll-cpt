"""Rebuild the per-dataset sample/tile plans (prepared/<id>/) of the frozen Studio run without touching data.json.

The run's prepared/ folders were deleted from the host after training. prepare.py `select` regenerates them but also
rewrites data.json; this wrapper calls the same select_one() for every dataset of the frozen label snapshot and
compares each result with the frozen data.json row (samples, distinct patches, tiles, shape, supervised ink and
background, eligible-before-holdout, holdout exclusion count, exclusion reason). It writes only prepared/<id>/
(labels.npy, coordinates.npy, entry-to-patch.npy, tiles.npy, selection.json) and rebuild/select-report.json.

  python rebuild_select.py [--workers N]
"""
import argparse, json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import prepare

KEYS = ('samples', 'distinct_patches', 'tiles', 'shape', 'supervised_ink', 'supervised_background', 'eligible_before_holdout',
        'exact_alias_pixels_removed', 'excluded_reason')


def one(row):
    try:
        return prepare.select_one(row)
    except Exception as exc:  # report, keep going
        return dict(id=row['id'], error=repr(exc)[:500])


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--workers', type=int, default=12); a = ap.parse_args()
    frozen = {r['id']: r for r in prepare.read(prepare.ROOT / 'data.json')['segments']}
    snapshot = prepare.read(prepare.ROOT / 'label-snapshot.json')
    with ProcessPoolExecutor(a.workers) as pool:
        rows = list(pool.map(one, snapshot['datasets']))
    report = []
    for r in rows:
        f = frozen.get(r['id'], {}); diff = {}
        if 'error' in r: diff['error'] = r['error']
        for k in KEYS:
            if r.get(k) != f.get(k): diff[k] = [f.get(k), r.get(k)]
        hf, hr = (f.get('holdout_exclusion') or {}).get('excluded_entries'), (r.get('holdout_exclusion') or {}).get('excluded_entries')
        if hf != hr: diff['holdout_excluded_entries'] = [hf, hr]
        report.append(dict(id=r['id'], match=not diff, diff=diff))
    out = prepare.ROOT / 'rebuild'; out.mkdir(exist_ok=True)
    (out / 'select-report.json').write_text(json.dumps(report, indent=1))
    ok = sum(x['match'] for x in report)
    print('SELECT_REBUILT', ok, 'of', len(report), 'match the frozen data.json', flush=True)
    for x in report:
        if not x['match']: print('MISMATCH', x['id'], json.dumps(x['diff'])[:400], flush=True)


if __name__ == '__main__':
    main()
