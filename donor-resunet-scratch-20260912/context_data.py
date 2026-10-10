"""Full donor pool and measured context for the scratch residual U-Net."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
from collections import Counter
import concurrent.futures as cf
import importlib.util
import json
import math
from pathlib import Path
import sys
import numpy as np

ROOT = Path(__file__).resolve().parent
ORIGINAL = ROOT.parent / 'donor-original-recipe-20260910'
DONOR = ROOT.parent / 'donor-continued-training-20260910'
SCROLLS = ['PHerc0814', 'PHerc0841', 'PHerc0009B']
SEED = 2026091201


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def old_tile_set(coordinates):
    return {(int(y)//128*128+dy, int(x)//128*128+dx)
        for y, x, _ in coordinates
        for dy in range(0, 128+(int(y)%128), 128)
        for dx in range(0, 128+(int(x)%128), 128)}


def needed_context_tiles(coordinates, shape):
    # Original random128 crop centers move +/-64px; V2 reads224px around them.
    # Therefore352px measured parent is sufficient (the old480 cache had64px
    # of unused outer context on each side for this particular128px recipe).
    h, w = shape
    needed = set()
    for y, x in coordinates:
        for yy in range(max(0, int(y)-112)//128*128, min(h, int(y)+240), 128):
            for xx in range(max(0, int(x)-112)//128*128, min(w, int(x)+240), 128):
                needed.add((yy, xx))
    return needed


def epoch_indices(doc, epoch):
    rng = np.random.default_rng(SEED + epoch)
    counts = doc['entry_counts']
    per_scroll = math.ceil(max(counts.values()) / 64) * 64
    draws = []
    for scroll in SCROLLS:
        pool = np.concatenate([np.arange(row['entry_offset'], row['entry_offset']+row['samples'])
            for row in doc['segments'] if row['role'] == 'train' and row['scroll'] == scroll])
        assert len(pool) == counts[scroll] and len(pool) <= per_scroll
        draws.append(np.concatenate([pool, rng.choice(pool, per_scroll-len(pool), replace=True)]))
    result = np.concatenate(draws)
    rng.shuffle(result)
    return result


def dry_run(doc):
    indices = epoch_indices(doc, 0)
    lookup = np.empty(doc['training_entries'], dtype=np.uint8)
    for row in doc['segments']:
        if row['role'] == 'train':
            lookup[row['entry_offset']:row['entry_offset']+row['samples']] = SCROLLS.index(row['scroll'])
    measured = Counter(lookup[indices].tolist())
    assert len(np.unique(indices)) == doc['training_entries']
    assert len(set(measured.values())) == 1
    report = dict(initialization='random weights; no pretrained checkpoint', s1_samples=0,
        supervision_mask=False, class_balancing=False, epoch_draws=len(indices),
        updates_per_epoch=len(indices)//64, all_original_training_entries_present=True,
        scrolls=[dict(scroll=s, eligible_entries=doc['entry_counts'][s],
            distinct_patches=doc['distinct_counts'][s], draws=measured[i], percent=100*measured[i]/len(indices))
            for i, s in enumerate(SCROLLS)],
        measured_parent=352, augmentation_fetch=224, model_window=128,
        missing_native_context_tiles=sum(r['new_context_tiles'] for r in doc['segments']),
        validation_id=doc['validation_id'])
    atomic(ROOT / 'dry-run.json', report)
    print(json.dumps(report, indent=2), flush=True)
    return report


def select():
    original = json.loads((ORIGINAL / 'dataset.json').read_text())
    rows = []
    entries = distinct = 0
    for source in original['segments']:
        assert source['scroll'] in SCROLLS, 'No S1 data may enter scratch training'
        if not source['samples']:
            continue
        old_folder = Path(source['folder'])
        coordinates = np.load(old_folder / 'coordinates.npy')
        assert len(coordinates) == source['samples']
        unique, inverse = np.unique(coordinates[:, :2], axis=0, return_inverse=True)
        folder = ROOT / 'context' / source['id']
        folder.mkdir(parents=True, exist_ok=True)
        np.save(folder / 'coordinates.npy', unique)
        np.save(folder / 'entry-to-patch.npy', inverse)
        base_tiles = old_tile_set(np.load(DONOR / 'prepared' / source['id'] / 'coordinates.npy'))
        extra = np.load(old_folder / 'extra-tile-coordinates.npy')
        known = base_tiles | {tuple(map(int, c)) for c in extra}
        # Validation is evaluated at its original fixed128px centers.
        required = needed_context_tiles(unique, source['shape']) if source['role'] == 'train' else old_tile_set(coordinates)
        missing = sorted(required - known)
        np.save(folder / 'extra-tile-coordinates.npy', np.asarray(missing, dtype=np.int32).reshape(-1, 2))
        row = dict(source, original_folder=str(old_folder), context_folder=str(folder),
            distinct_patches=len(unique), new_context_tiles=len(missing))
        if source['role'] == 'train':
            row.update(entry_offset=entries, patch_offset=distinct)
            entries += len(coordinates)
            distinct += len(unique)
        rows.append(row)
    counts = {s: sum(r['samples'] for r in rows if r['role'] == 'train' and r['scroll'] == s) for s in SCROLLS}
    unique_counts = {s: sum(r['distinct_patches'] for r in rows if r['role'] == 'train' and r['scroll'] == s) for s in SCROLLS}
    doc = dict(status='selected', initialization='scratch', source_dataset=str(ORIGINAL / 'dataset.json'),
        validation_id=original['validation_id'], validation_overlap_exclusion_um=original['validation_overlap_exclusion_um'],
        entry_counts=counts, distinct_counts=unique_counts, training_entries=entries,
        distinct_training_patches=distinct, segments=rows,
        pool_policy='Every original eligible donor entry, including multiplicity; each appears every epoch; equal scroll totals with replacement topups',
        context_policy='352px measured parent centered on each128px reference; original +/-64 random crop and224px V2 augmentation context',
        supervision_mask=False, class_balancing=False, s1_samples=0)
    atomic(ROOT / 'data.json', doc)
    dry_run(doc)


def render():
    doc = json.loads((ROOT / 'data.json').read_text())
    spec = importlib.util.spec_from_file_location('accepted_native_context', DONOR / 'prepare.py')
    native = importlib.util.module_from_spec(spec)
    sys.path.insert(0, str(DONOR))
    spec.loader.exec_module(native)
    catalog = json.loads((DONOR / 'volume-catalog.json').read_text())
    volumes = {}
    for row in doc['segments']:
        folder = Path(row['context_folder'])
        if (folder / 'ready.json').exists():
            continue
        tiles = np.load(folder / 'extra-tile-coordinates.npy')
        if len(tiles):
            output = folder / 'extra-ct.npy'
            ct = np.lib.format.open_memmap(output, mode='r+' if output.exists() else 'w+',
                dtype=np.uint8, shape=(len(tiles), 16, 128, 128))
            surface, recipe = native.surface_for(row, catalog)
            if surface is not None:
                if recipe['url'] not in volumes:
                    for volume in volumes.values():
                        volume.pool.shutdown(wait=True)
                    volumes.clear()
                    volumes[recipe['url']] = native.FastVolume(recipe['url'], cache_gb=12, workers=8)
                volume = volumes[recipe['url']]
            else:
                source = ROOT.parent / 'donor-labels-20260906/prepared' / row['id'] / 'ct.npy'
                if not source.exists() and row['id'].startswith('source-'):
                    source = ROOT.parent / 'donor-labels-20260906/prepared' / row['id'].removeprefix('source-').split('-approved-')[0] / 'ct.npy'
                if not source.exists():
                    source = DONOR / 'legacy-ct' / row['id'] / 'ct.npy'
                old = np.load(source, mmap_mode='r')
                old = old[len(old)//2-8:len(old)//2+8]
                if (row.get('orientation') or {}).get('normal_sign') == -1:
                    old = old[::-1]
            progress = folder / 'progress.json'
            done = json.loads(progress.read_text())['done'] if progress.exists() else 0

            def one(item):
                i, (y, x) = item
                if surface is not None:
                    a, valid = surface.render(volume, np.arange(y, y+128), np.arange(x, x+128),
                        recipe['xy_pitch_um'], recipe['offsets_um'])
                    a[:, ~valid] = 0
                else:
                    a = np.zeros((16, 128, 128), dtype=np.uint8)
                    block = old[:, y:min(y+128, old.shape[1]), x:min(x+128, old.shape[2])]
                    a[:, :block.shape[1], :block.shape[2]] = block
                # Match existing caches: runtime_reverse is applied ONCE at read time.
                return i, a

            with cf.ThreadPoolExecutor(max_workers=2) as pool:
                for i, values in native.bounded_map(pool, one, list(enumerate(tiles))[done:], 4):
                    ct[i] = values
                    if (i+1) % 32 == 0 or i+1 == len(tiles):
                        ct.flush()
                        atomic(progress, dict(done=i+1, total=len(tiles)))
                        print('CONTEXT', row['id'], i+1, '/', len(tiles), flush=True)
            ct.flush()
        atomic(folder / 'ready.json', dict(status='complete', extra_tiles=len(tiles),
            runtime_reverse_at_read=bool(row.get('runtime_reverse'))))
    for volume in volumes.values():
        volume.pool.shutdown(wait=True)
    doc['status'] = 'complete'
    atomic(ROOT / 'data.json', doc)
    print('ALL_CONTEXT_READY', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=['select', 'render', 'dry-run'])
    args = parser.parse_args()
    if args.action == 'select':
        select()
    elif args.action == 'render':
        render()
    else:
        dry_run(json.loads((ROOT / 'data.json').read_text()))
