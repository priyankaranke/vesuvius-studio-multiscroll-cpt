"""Original tile eligibility on donor labels; reuse native CT and render missing tiles."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import argparse
import concurrent.futures as cf
import json
from pathlib import Path
import sys

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from PIL import Image
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parent
PREVIOUS = ROOT.parent / 'donor-continued-training-20260910'
sys.path.insert(0, str(PREVIOUS))
import importlib.util
spec = importlib.util.spec_from_file_location('native_preparation', PREVIOUS / 'prepare.py')
native = importlib.util.module_from_spec(spec)
spec.loader.exec_module(native)
Image.MAX_IMAGE_PIXELS = None


def atomic(path, value):
    tmp = path.with_suffix(path.suffix + '.tmp')
    tmp.write_text(json.dumps(value, indent=2))
    tmp.replace(path)


def coordinates(ink, physical_mask, validation=False):
    """Exactly original 256/64 parents and four 128 children, including multiplicity."""
    h, w = ink.shape
    ph, pw = (h + 255) // 256 * 256, (w + 255) // 256 * 256
    ink = np.pad(ink, ((0, ph - h), (0, pw - w)))
    mask = np.pad(physical_mask, ((0, ph - h), (0, pw - w)))
    blocks = ink.reshape(ph // 64, 64, pw // 64, 64)
    has_ink = blocks.any((1, 3))
    valid = mask.reshape(ph // 64, 64, pw // 64, 64).all((1, 3))
    parents = sliding_window_view(valid, (4, 4)).all((-1, -2))
    if not validation:
        parents &= sliding_window_view(has_ink, (4, 4)).any((-1, -2))
    children = sliding_window_view(has_ink, (2, 2)).any((-1, -2))
    rows = []
    seen = set()
    for y, x in zip(*np.nonzero(parents)):
        for dy in [0, 2]:
            for dx in [0, 2]:
                yy, xx = int(y + dy), int(x + dx)
                if validation and (yy, xx) in seen:
                    continue
                seen.add((yy, xx))
                rows.append((yy * 64, xx * 64, int(children[yy, xx])))
    return np.asarray(rows, dtype=np.int32).reshape(-1, 3)


def tile_set(coords):
    return {(int(y)//128*128+dy, int(x)//128*128+dx)
            for y, x, _ in coords
            for dy in range(0, 128+(int(y)%128), 128)
            for dx in range(0, 128+(int(x)%128), 128)}


def covered_ink(ink, coords):
    h, w = ink.shape
    blocks = np.zeros(((h+63)//64, (w+63)//64), dtype=bool)
    for y, x, _ in coords:
        blocks[y//64:y//64+2, x//64:x//64+2] = True
    covered = np.repeat(np.repeat(blocks, 64, axis=0), 64, axis=1)[:h, :w]
    return int((ink & covered).sum())


def select():
    rows = json.loads((PREVIOUS / 'labels.json').read_text())
    previous = json.loads((PREVIOUS / 'dataset.json').read_text())
    catalog = json.loads((PREVIOUS / 'volume-catalog.json').read_text())
    validation_id = previous['validation_id']
    holdout = next(r for r in rows if r['id'] == validation_id)
    surface, _ = native.surface_for(holdout, catalog, native=False)
    tree = cKDTree(surface.a[surface.valid] * surface.ct_um)
    output = []
    for row in rows:
        out = ROOT / 'prepared' / row['id']
        out.mkdir(parents=True, exist_ok=True)
        labels = PREVIOUS / row['label_directory']
        # Painted/unpainted supervision is deliberately never read.
        ink = np.asarray(Image.open(labels / 'inklabels.png')) >= 13
        physical = np.asarray(Image.open(labels / 'geometry-valid.png')) > 0
        coords = coordinates(ink, physical, row['id'] == validation_id)
        eligible, before_holdout = len(coords), covered_ink(ink, coords)
        if row['scroll'] == holdout['scroll'] and row['id'] != validation_id:
            s, _ = native.surface_for(row, catalog, native=False)
            if s is None:
                coords = coords[:0]
            elif len(coords):
                unique, inverse = np.unique(coords[:, :2], axis=0, return_inverse=True)
                centers = []
                for y, x in unique:
                    p, _, v = s.points([int(y)+64], [int(x)+64], row['source_meta']['target_um'])
                    centers.append(p[0, 0]*s.ct_um if v[0, 0] else np.full(3, np.inf))
                centers = np.asarray(centers)
                finite = np.isfinite(centers).all(1)
                keep = np.zeros(len(unique), dtype=bool)
                keep[finite] = tree.query(centers[finite], workers=4)[0] > previous['validation_overlap_exclusion_um']
                coords = coords[keep[inverse]]
        np.save(out / 'coordinates.npy', coords)
        old = PREVIOUS / 'prepared' / row['id']
        ready = json.loads((old / 'prepared.json').read_text()) if (old / 'prepared.json').exists() else {}
        old_tiles = tile_set(np.load(old / 'coordinates.npy')) if ready else set()
        missing = sorted(tile_set(coords) - old_tiles)
        np.save(out / 'extra-tile-coordinates.npy', np.asarray(missing, dtype=np.int32).reshape(-1, 2))
        result = dict(row, role='validation' if row['id'] == validation_id else 'train',
                      samples=len(coords), positive_samples=int(coords[:, 2].sum()),
                      original_eligible_entries=eligible, excluded_overlap_samples=eligible-len(coords),
                      labeled_ink_pixels=int(ink.sum()), ink_pixels_original_eligible=before_holdout,
                      ink_pixels_retained=covered_ink(ink, coords),
                      folder=str(out), base_ct_path=str(old / 'ct.npy'), extra_tiles=len(missing),
                      runtime_reverse=ready.get('runtime_reverse', False),
                      orientation_resolved=ready.get('orientation_resolved', False),
                      recipe=ready.get('recipe'), depth_calibration=ready.get('depth_calibration'))
        atomic(out / 'selection.json', result)
        output.append(result)
        print('SELECT', row['id'], len(coords), 'extra_tiles', len(missing), flush=True)
    atomic(ROOT / 'dataset.json', dict(seed=0, validation_id=validation_id,
           validation_selection_seed=previous['seed'],
           validation_overlap_exclusion_um=previous['validation_overlap_exclusion_um'],
           eligibility='Original full physical 256px parent support, stride64, any ink in training parent, all four128 children. No supervision mask or intensity filter.',
           segments=output))


def render():
    doc = json.loads((ROOT / 'dataset.json').read_text())
    catalog = json.loads((PREVIOUS / 'volume-catalog.json').read_text())
    volumes = {}
    for row in doc['segments']:
        out = Path(row['folder'])
        if not row['samples'] or (out / 'prepared.json').exists():
            continue
        if not row['orientation_resolved']:
            raise RuntimeError('Unresolved orientation: ' + row['id'])
        tiles = np.load(out / 'extra-tile-coordinates.npy')
        if len(tiles):
            path = out / 'extra-ct.npy'
            ct = np.lib.format.open_memmap(path, mode='r+' if path.exists() else 'w+',
                                          dtype=np.uint8, shape=(len(tiles), 16, 128, 128))
            s, recipe = native.surface_for(row, catalog)
            if s is None:
                # These tiles come from the original fully measured 65-plane donor cache.
                old = ROOT.parent / 'donor-labels-20260906/prepared' / row['id'] / 'ct.npy'
                if not old.exists() and row['id'].startswith('source-'):
                    old = ROOT.parent / 'donor-labels-20260906/prepared' / row['id'].removeprefix('source-').split('-approved-')[0] / 'ct.npy'
                if not old.exists():
                    old = PREVIOUS / 'legacy-ct' / row['id'] / 'ct.npy'
                source = np.load(old, mmap_mode='r')
                source = source[len(source)//2-8:len(source)//2+8]
                if (row.get('orientation') or {}).get('normal_sign') == -1:
                    source = source[::-1]
                for i, (y, x) in enumerate(tiles):
                    ct[i] = source[:, y:y+128, x:x+128]
            else:
                if recipe['url'] not in volumes:
                    for volume in volumes.values():
                        volume.pool.shutdown(wait=True)
                    volumes.clear()
                    volumes[recipe['url']] = native.FastVolume(recipe['url'], cache_gb=12, workers=16)
                volume = volumes[recipe['url']]
                progress = out / 'render-progress.json'
                done = json.loads(progress.read_text())['done'] if progress.exists() else 0
                def one(item):
                    i, (y, x) = item
                    a, valid = s.render(volume, np.arange(y, y+128), np.arange(x, x+128),
                                        recipe['xy_pitch_um'], recipe['offsets_um'])
                    a[:, ~valid] = 0
                    return i, a
                with cf.ThreadPoolExecutor(4) as pool:
                    for i, a in native.bounded_map(pool, one, list(enumerate(tiles))[done:], 8):
                        ct[i] = a
                        if (i+1) % 32 == 0 or i+1 == len(tiles):
                            ct.flush()
                            atomic(progress, dict(done=i+1, total=len(tiles)))
                            print('RENDER', row['id'], i+1, '/', len(tiles), flush=True)
            ct.flush()
        atomic(out / 'prepared.json', dict(row, status='complete'))
        print('READY', row['id'], row['samples'], flush=True)
    for volume in volumes.values():
        volume.pool.shutdown(wait=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=['select', 'render'])
    args = parser.parse_args()
    select() if args.stage == 'select' else render()
