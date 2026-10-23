"""Host-RAM tile atlas for the non-Studio scrolls, in the Expanded9 prepared-row format.

A prepared row folder holds
  tiles.npy          int [N,2]  top-left (y, x) of each 128x128 CT tile on the row's canvas (multiples of 128)
  ct-tiles.blosc     the N Blosc-compressed uint8 (128,128,16) tiles (Y, X, depth), concatenated in tiles.npy order,
  ct-tile-offsets.npy  uint64 [N+1] byte offsets into ct-tiles.blosc        (Expanded9 HF layout)
    or ct-tiles.npy  uint8 [N,128,128,16]                                    (rows prepared here, e.g. Reader v2 labels)
  coordinates.npy    int [S,2]  top-left (y, x) of each 128 px training patch (the 352 px parent around it is prepared)
  label.png          uint8 canvas, ink >= 128 (soft values in between are left out of the loss, see below)
  mask.png           uint8 canvas, > 0 where the label is valid (papyrus / supervision mask)
  ready.json         row metadata (id, scroll, role, shape, samples, pitch_um, source, ...)

Labels become Studio's categorical state: 0 unknown (outside the mask, or a soft label strictly between 64 and 128),
1 background (<= 64), 2 ink (>= 128). CT and state crops follow the Studio/Expanded9 sampler exactly: a 224 px context
at (y - 112 + offset, x - 112 + offset), offset in [0, 128], reflected CT and zero (unknown) state outside the canvas.
"""
import concurrent.futures as cf, json, os
from pathlib import Path
import numpy as np, numcodecs
from PIL import Image

Image.MAX_IMAGE_PIXELS = None
numcodecs.blosc.set_nthreads(1)


def categorical(label, mask):
    state = np.zeros(label.shape, np.uint8)
    valid = mask > 0
    state[valid & (label <= 64)] = 1
    state[valid & (label >= 128)] = 2
    return state


class HostAtlas:
    def __init__(self, folders, workers=32):
        self.pool = cf.ThreadPoolExecutor(workers); self.rows = []
        metas = [json.loads((Path(f) / 'ready.json').read_text()) for f in folders]
        count = sum(len(np.load(Path(f) / 'tiles.npy')) for f in folders)
        self.atlas = np.empty((count, 128, 128, 16), np.uint8); start = 0
        for folder, meta in zip(folders, metas):
            folder = Path(folder); tiles = np.load(folder / 'tiles.npy').astype(np.int64)
            if (folder / 'ct-tiles.npy').exists():
                values = np.load(folder / 'ct-tiles.npy', mmap_mode='r'); assert values.shape == (len(tiles), 128, 128, 16)
                self.atlas[start:start + len(tiles)] = values
            else:
                blob = (folder / 'ct-tiles.blosc').read_bytes(); offsets = np.load(folder / 'ct-tile-offsets.npy').astype(np.int64)
                assert len(offsets) == len(tiles) + 1 and offsets[-1] == len(blob)

                def decode(i, blob=blob, offsets=offsets, start=start):
                    a = numcodecs.blosc.decompress(blob[offsets[i]:offsets[i + 1]])
                    self.atlas[start + i] = np.frombuffer(a, np.uint8).reshape(128, 128, 16)
                list(self.pool.map(decode, range(len(tiles))))
                del blob
            label = np.asarray(Image.open(folder / 'label.png')); mask = np.asarray(Image.open(folder / 'mask.png'))
            if label.ndim == 3: label = label[..., 0]
            if mask.ndim == 3: mask = mask[..., 0]
            h, w = label.shape; assert tuple(meta['shape']) == (h, w) and mask.shape == (h, w)
            grid = np.full(((h + 127) // 128, (w + 127) // 128), -1, np.int64)
            grid[tiles[:, 0] // 128, tiles[:, 1] // 128] = np.arange(start, start + len(tiles))
            coords = np.load(folder / 'coordinates.npy').astype(np.int64)
            state = categorical(label, mask)
            self.rows.append(dict(meta=meta, id=meta['id'], scroll=meta['scroll'], grid=grid, state=state, coords=coords,
                                  ink=int((state == 2).sum()), background=int((state == 1).sum())))
            start += len(tiles)
            print('HOST_ATLAS', meta['id'], meta['scroll'], len(tiles), 'tiles', len(coords), 'samples', flush=True)
        self.bytes = int(self.atlas.nbytes + sum(r['state'].nbytes for r in self.rows))

    def crop(self, ri, y, x, width=224):
        """Expanded9's Atlas.crop: canvas-clipped tiles, then reflected CT and zero (unknown) state padding."""
        r = self.rows[ri]; state = r['state']; h, w = state.shape
        y1, y2, x1, x2 = max(0, y), min(h, y + width), max(0, x), min(w, x + width)
        a = np.empty((y2 - y1, x2 - x1, 16), np.uint8)
        for yy in range(y1 // 128 * 128, y2, 128):
            for xx in range(x1 // 128 * 128, x2, 128):
                index = r['grid'][yy // 128, xx // 128]
                if index < 0:
                    raise ValueError(('Unprepared context tile', r['id'], yy, xx))
                b, c = max(y1, yy), min(y2, yy + 128); d, e = max(x1, xx), min(x2, xx + 128)
                a[b - y1:c - y1, d - x1:e - x1] = self.atlas[index, b - yy:c - yy, d - xx:e - xx]
        target = state[y1:y2, x1:x2]
        padding = ((max(0, -y), max(0, y + width - h)), (max(0, -x), max(0, x + width - w)))
        if any(v for pair in padding for v in pair):
            a = np.pad(a, (*padding, (0, 0)), mode='reflect'); target = np.pad(target, padding, mode='constant')
        return a, target

    def fill(self, items, raw_out, state_out):
        """items: list of (row index, patch index, offset_y, offset_x)."""
        def one(i):
            ri, pi, oy, ox = items[i]; y, x = self.rows[ri]['coords'][pi]
            raw_out[i], state_out[i] = self.crop(ri, int(y) - 112 + int(oy), int(x) - 112 + int(ox))
        list(self.pool.map(one, range(len(items))))

    def close(self):
        self.pool.shutdown()


class GroupStream:
    """Deterministic without-replacement cycling over every training patch of a scroll group (all its rows pooled)."""
    def __init__(self, entries, seed):
        self.entries = entries; self.seed = seed; self.cycle = 0; self.position = 0; self._perm()

    def _perm(self):
        self.order = np.random.default_rng(np.random.SeedSequence([self.seed, self.cycle])).permutation(len(self.entries))

    def take(self, n):
        out = []
        while len(out) < n:
            if self.position == len(self.order):
                self.cycle += 1; self.position = 0; self._perm()
            k = min(n - len(out), len(self.order) - self.position)
            out.extend(self.entries[self.order[self.position:self.position + k]]); self.position += k
        return out

    def state(self):
        return dict(cycle=self.cycle, position=self.position)

    def load(self, s):
        self.cycle, self.position = s['cycle'], s['position']; self._perm()
