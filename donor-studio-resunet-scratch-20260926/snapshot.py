"""Freeze committed Studio labels without writing to any production dataset."""
import collections
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import time

ROOT = Path(__file__).resolve().parent
STUDIO = ROOT.parent / 'donor-spirals-20260921'


def freeze():
    destination = ROOT / 'labels'
    configuration = ROOT / 'label-snapshot.json'
    if configuration.exists():
        print(configuration.read_text())
        return
    # Exclude the in-progress wall-clock second. All later commits remain outside
    # this run even if publication/editing continues during the backup.
    cutoff = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime(time.time()-1))
    rows = []
    for scroll in ('814', '841', '9B'):
        for category in ('full-canvases', 'hr-rois', 'original-labels'):
            for path in sorted((STUDIO / scroll / category).glob('*/dataset.json')):
                source = path.parent
                meta = json.loads(path.read_text())
                row = dict(id=meta['id'], scroll=scroll, category=category,
                           source_directory=str(source), metadata=meta)
                target = destination / category / meta['id']
                files = ['dataset.json']
                if category == 'full-canvases':
                    if not (source / 'labels.sqlite3').exists():
                        continue
                    with sqlite3.connect(f'file:{source / "labels.sqlite3"}?mode=ro', uri=True) as db:
                        db.execute('BEGIN')
                        rev = db.execute('SELECT id,time,ink,background,orientation FROM revisions WHERE time<=? ORDER BY id DESC LIMIT 1', (cutoff,)).fetchone()
                        if not rev or not (rev[2] or rev[3]):
                            continue
                        target.mkdir(parents=True, exist_ok=True)
                        with sqlite3.connect(target / 'labels.sqlite3') as out:
                            db.backup(out)
                        row.update(revision=rev[0], revision_time=rev[1], ink=rev[2], background=rev[3],
                                   orientation=json.loads(rev[4]), labels_file='labels.sqlite3',
                                   encoding={'0': 'unknown', '1': 'background', '2': 'ink'})
                    files += ['mapping.json', meta['geometry_file']]
                elif category == 'hr-rois':
                    revisions = [(int(p.stem[1:]), json.loads(p.read_text()), p) for p in (source / 'revisions').glob('r*.json')]
                    revisions = [r for r in revisions if r[1].get('time_utc', '') <= cutoff and r[2].with_suffix('.png').exists()]
                    rev, record, revision_path = max(revisions, default=(0, {}, None), key=lambda r: r[0])
                    state = revision_path.with_suffix('.png') if rev else source / 'source-state.png'
                    if not state.exists():
                        continue
                    from PIL import Image
                    import numpy as np
                    a = np.asarray(Image.open(state))
                    values = set(np.unique(a).tolist())
                    if not values <= {0, 1, 2}:
                        raise ValueError(('Unexpected ROI encoding', str(state), values))
                    ink, background = int((a == 2).sum()), int((a == 1).sum())
                    if not ink + background:
                        continue
                    target.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(state, target / 'labels.png')
                    if rev:
                        shutil.copy2(revision_path, target / 'revision.json')
                    row.update(revision=rev, revision_time=record.get('time_utc'), ink=ink, background=background,
                               orientation=record.get('training_orientation', meta.get('training_orientation')),
                               labels_file='labels.png', encoding={'0': 'unknown', '1': 'background', '2': 'ink'})
                    files += ['roi.json', meta.get('geometry_file', 'geometry-valid.png')]
                else:
                    if not meta['inkPixels'] + meta['backgroundPixels']:
                        continue
                    target.mkdir(parents=True, exist_ok=True)
                    row.update(revision=meta['revision'], revision_time=meta.get('sourceRevisionTime'),
                               ink=meta['inkPixels'], background=meta['backgroundPixels'],
                               orientation=meta.get('labelingOrientation'), labels_file=meta['labelsFile'],
                               encoding={'0': 'unknown', '1': 'background', '2': 'ink'},
                               label_provenance=meta.get('labelSource'))
                    files += [meta['sourceMetadataFile'], meta['historyFile'], meta['labelsFile'], meta['validMaskFile']]
                    if meta.get('validationFile'):
                        files.append(meta['validationFile'])
                for filename in dict.fromkeys(files):
                    shutil.copy2(source / filename, target / filename)
                row['snapshot_directory'] = str(target)
                row['sha256'] = {name: hashlib.sha256((target / name).read_bytes()).hexdigest()
                                 for name in dict.fromkeys(files + [row['labels_file']])}
                rows.append(row)
                print('FROZEN', category, row['id'], row['revision'], row['ink'], row['background'], flush=True)
    result = dict(cutoff_utc=cutoff, source='live Studio committed states', encoding='0 unknown, 1 background, 2 ink',
                  counts=dict(collections.Counter(r['category'] for r in rows)),
                  scroll_counts=dict(collections.Counter(r['scroll'] for r in rows)),
                  datasets=rows, visibility_policy='Include labeled hidden datasets; visibility is not exclusion')
    temporary = configuration.with_suffix('.tmp')
    temporary.write_text(json.dumps(result, indent=2)+'\n')
    temporary.replace(configuration)
    print('SNAPSHOT_COMPLETE', cutoff, len(rows), result['counts'], result['scroll_counts'], flush=True)


if __name__ == '__main__':
    freeze()
