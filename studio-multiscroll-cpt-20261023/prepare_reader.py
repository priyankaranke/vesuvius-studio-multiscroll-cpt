"""Prepare Reader v2/v3 training labels (HF dataset domenicor046/reader-v2-labels, MIT) as host-atlas rows.

Reader V3 kept Reader v2's training data and labels, so these are the labels behind both models: dense ink maps of a
finer scan, resampled onto the team's native 8-9 um surface volumes (ink = map >= 128, supervised = map >= 128 or <= 64;
labels on the middle layer). Here, for every segment of the chosen objects:
  * label plane z = label_z of `<seg>_inklabels.zarr` and `<seg>_supervision_mask.zarr` -> label.png / mask.png;
  * training patches: 128 px cells on a 128 px grid whose central 96 px is >= 25 % supervised; cells with >= 5 % ink in
    their core get a second entry (positives counted twice, like the Expanded9 samplers);
  * CT: the public native surface volume (s3 key in manifest.csv, zarr level 0, uncompressed (D,128,128) chunks), only
    the chunks covering each patch's 352 px parent context, planes label_z-8 .. label_z+7 in their stored order
    (Studio's 814-native rows read the team's 28-layer volumes as planes 6:22 the same way; check_orientation.py
    verifies the order against Studio before use).

  python prepare_reader.py --labels reader-v2/reader-v2-labels --manifest reader-v2/manifest.csv --out reader-rows \
      --objects 0139 0500P2 0343P 0172 [--max-patches 0172=700]
"""
import argparse, concurrent.futures as cf, csv, hashlib, json, os, time
from pathlib import Path
import numpy as np, numcodecs, requests
from PIL import Image

BASE = 'https://vesuvius-challenge-open-data.s3.amazonaws.com/'
SCROLL = {'0139': 'PHerc0139', '0500P2': 'PHerc0500P2', '0343P': 'PHerc0343P', '0172': 'PHerc0172', '0814': 'PHerc0814', '0009B': 'PHerc0009B'}
session = requests.Session(); session.mount('https://', requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64))


def label_plane(zarr_dir, z, shape):
    meta = json.loads((zarr_dir / '0' / '.zarray').read_text()); assert meta['chunks'][1:] == [128, 128] and meta['dimension_separator'] == '.'
    d, h, w = shape; out = np.zeros((h, w), np.uint8); codec = numcodecs.get_codec(meta['compressor']) if meta['compressor'] else None
    for f in (zarr_dir / '0').iterdir():
        if f.name.startswith('.'): continue
        cz, cy, cx = map(int, f.name.split('.')); assert cz == 0
        b = f.read_bytes(); b = codec.decode(b) if codec else b
        a = np.frombuffer(b, np.uint8).reshape(meta['chunks'])[z]
        hh, ww = min(128, h - cy * 128), min(128, w - cx * 128); out[cy * 128:cy * 128 + hh, cx * 128:cx * 128 + ww] = a[:hh, :ww]
    return out


def fetch_chunk(url, depth, retries=6):
    for attempt in range(retries):
        try:
            r = session.get(url, timeout=(20, 120))
            if r.status_code in (403, 404): return None
            r.raise_for_status(); b = r.content; assert len(b) == depth * 128 * 128, (url, len(b)); return b
        except Exception:
            if attempt == retries - 1: raise
            time.sleep(2 * (attempt + 1))


def prepare(row, labels_root, out_root, max_patches, workers, seed=2026102304):
    obj, seg = row['object'], row['segment']; d, h, w = map(int, row['canvas_zyx'].split('x')); lz = int(row['label_z'])
    out = out_root / f'reader-{obj}-{seg}'
    if (out / 'ready.json').exists(): return json.loads((out / 'ready.json').read_text())
    out.mkdir(parents=True, exist_ok=True); began = time.monotonic()
    folder = labels_root / obj / seg
    ink = label_plane(folder / f'{seg}_inklabels.zarr', lz, (d, h, w)) > 127
    sup = label_plane(folder / f'{seg}_supervision_mask.zarr', lz, (d, h, w)) > 127
    state = np.where(sup, np.where(ink, 2, 1), 0).astype(np.uint8)
    cells = []
    for y in range(0, h - 127, 128):
        for x in range(0, w - 127, 128):
            core = state[y + 16:y + 112, x + 16:x + 112]; s = core > 0
            if s.mean() >= .25: cells.append((y, x, float((core == 2).sum() / max(1, s.sum()))))
    rng = np.random.default_rng([seed, int.from_bytes(hashlib.sha256(seg.encode()).digest()[:4], 'little')])
    if max_patches and len(cells) > max_patches:
        cells = [cells[i] for i in sorted(rng.choice(len(cells), max_patches, replace=False))]
    coords = [(y, x) for y, x, f in cells] + [(y, x) for y, x, f in cells if f >= .05]
    if not coords:
        (out / 'ready.json').write_text(json.dumps(dict(id=out.name, status='empty'))); return None
    tiles = set()
    for y, x, _ in cells:
        for ty in range(max(0, y - 112) // 128, min(h - 1, y + 239) // 128 + 1):
            for tx in range(max(0, x - 112) // 128, min(w - 1, x + 239) // 128 + 1): tiles.add((ty, tx))
    tiles = sorted(tiles); key = row['surface_volume_s3'].rstrip('/')
    meta = json.loads(session.get(BASE + key + '/0/.zarray', timeout=60).content)
    assert meta['shape'] == [d, h, w] and meta['chunks'] == [d, 128, 128] and meta['compressor'] is None and meta['dtype'] == '|u1', meta
    sep = meta.get('dimension_separator', '.'); planes = slice(lz - 8, lz + 8)
    ct = np.lib.format.open_memmap(out / 'ct-tiles.partial.npy', mode='w+', dtype=np.uint8, shape=(len(tiles), 128, 128, 16))
    missing = []

    def one(i):
        ty, tx = tiles[i]; b = fetch_chunk(BASE + key + '/0/' + sep.join(['0', str(ty), str(tx)]), d)
        if b is None: missing.append((ty, tx)); ct[i] = 0; return
        ct[i] = np.frombuffer(b, np.uint8).reshape(d, 128, 128)[planes].transpose(1, 2, 0)
    with cf.ThreadPoolExecutor(workers) as pool: list(pool.map(one, range(len(tiles))))
    ct.flush(); del ct; (out / 'ct-tiles.partial.npy').replace(out / 'ct-tiles.npy')
    np.save(out / 'tiles.npy', np.array([(ty * 128, tx * 128) for ty, tx in tiles], np.int64))
    np.save(out / 'coordinates.npy', np.array(coords, np.int64))
    Image.fromarray(ink.astype(np.uint8) * 255).save(out / 'label.png'); Image.fromarray(sup.astype(np.uint8) * 255).save(out / 'mask.png')
    ready = dict(id=out.name, scroll=SCROLL[obj], role='train', shape=[h, w], samples=len(coords), distinct_patches=len(cells), tiles=len(tiles),
                 source='reader-v2-labels (domenicor046/reader-v2-labels, MIT; labels behind Reader v2 and Reader V3)', segment=seg,
                 surface_volume=BASE + key, teacher_map=BASE + row['teacher_map_s3'], label_z=lz, planes=[lz - 8, lz + 7], canvas_zyx=[d, h, w],
                 missing_chunks=len(missing), supervised_pixels=int(sup.sum()), ink_pixels=int((ink & sup).sum()), status='complete',
                 seconds=round(time.monotonic() - began, 1))
    (out / 'ready.json').write_text(json.dumps(ready, indent=1)); print('READY', json.dumps({k: ready[k] for k in ('id', 'samples', 'tiles', 'missing_chunks', 'seconds')}), flush=True)
    return ready


def main():
    p = argparse.ArgumentParser(); p.add_argument('--labels', required=True); p.add_argument('--manifest', required=True); p.add_argument('--out', required=True)
    p.add_argument('--objects', nargs='+', default=['0139', '0500P2', '0343P', '0172']); p.add_argument('--max-patches', nargs='*', default=['0172=700'])
    p.add_argument('--workers', type=int, default=48); p.add_argument('--segments-in-parallel', type=int, default=3); a = p.parse_args()
    caps = {k: int(v) for k, v in (s.split('=') for s in a.max_patches)}
    rows = [r for r in csv.DictReader(open(a.manifest)) if r['object'] in a.objects]
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    with cf.ThreadPoolExecutor(a.segments_in_parallel) as pool:
        results = list(pool.map(lambda r: prepare(r, Path(a.labels), out, caps.get(r['object'], 0), a.workers), rows))
    print('ALL_READY', sum(1 for r in results if r), 'rows', flush=True)


if __name__ == '__main__':
    main()
