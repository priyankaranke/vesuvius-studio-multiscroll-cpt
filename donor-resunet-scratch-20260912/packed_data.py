"""Native donor contexts, equal-scroll epochs, and unchanged study augmentation."""
from pathlib import Path
import sys
import numpy as np
from PIL import Image
import torch
from context_data import ROOT, DONOR, SCROLLS, epoch_indices

STUDY = ROOT.parent / 'pherc1447-architecture-study-20260906'
sys.path.insert(0, str(STUDY))
from scroll_augment import augment as scroll_shift
Image.MAX_IMAGE_PIXELS = None


class NativeCanvas:
    def __init__(self, row):
        assert row['scroll'] in SCROLLS and row['orientation_resolved']
        self.row = row
        self.base = np.load(row['base_ct_path'], mmap_mode='r')
        self.label = np.asarray(Image.open(DONOR / row['label_directory'] / 'inklabels.png'), dtype=np.uint8)
        assert list(self.label.shape) == row['shape']
        self.extra = {}
        for key in ['original_folder', 'context_folder']:
            folder = Path(row[key])
            coords = np.load(folder / 'extra-tile-coordinates.npy')
            if len(coords):
                values = np.load(folder / 'extra-ct.npy', mmap_mode='r')
                assert values.shape == (len(coords), 16, 128, 128)
                self.extra.update({tuple(map(int, yx)): values[i] for i, yx in enumerate(coords)})

    def crop(self, y, x, width):
        h, w = self.label.shape
        y1, y2 = max(0, y), min(h, y+width)
        x1, x2 = max(0, x), min(w, x+width)
        image = self.base[:, y1:y2, x1:x2].copy()
        for yy in range(y1//128*128, y2, 128):
            for xx in range(x1//128*128, x2, 128):
                tile = self.extra.get((yy, xx))
                if tile is not None:
                    a, b = max(y1, yy), min(y2, yy+128)
                    c, d = max(x1, xx), min(x2, xx+128)
                    image[:, a-y1:b-y1, c-x1:d-x1] = tile[:, a-yy:b-yy, c-xx:d-xx]
        target = self.label[y1:y2, x1:x2]
        padding = ((max(0, -y), max(0, y+width-h)), (max(0, -x), max(0, x+width-w)))
        if any(v for pair in padding for v in pair):
            image = np.pad(image, ((0, 0), *padding), mode='reflect')
            target = np.pad(target, padding, mode='constant')
        if self.row.get('runtime_reverse'):
            image = image[::-1]
        assert image.shape == (16, width, width) and target.shape == (width, width)
        return image.transpose(1, 2, 0), target


def load_training(doc, keep_cpu=False):
    assert doc['status'] == 'complete' and doc['s1_samples'] == 0
    count = doc['distinct_training_patches']
    required = count * 352 * 352 * 17
    free, _ = torch.cuda.mem_get_info()
    # Accepted micro16 training peaks below27GiB. Preserve36GiB for model,
    # optimizer, activations, validation and bounded batch construction.
    device = 'cuda' if required + 36 * 2**30 < free else 'cpu'
    arena_device = 'cpu' if keep_cpu else device
    x = torch.empty((count, 352, 352, 16), dtype=torch.uint8, device=arena_device)
    y = torch.empty((count, 352, 352), dtype=torch.uint8, device=arena_device)
    refs = np.empty(doc['training_entries'], dtype=np.int64)
    for row in doc['segments']:
        if row['role'] != 'train':
            continue
        folder = Path(row['context_folder'])
        assert (folder / 'ready.json').exists()
        coords = np.load(folder / 'coordinates.npy')
        inverse = np.load(folder / 'entry-to-patch.npy')
        offset = row['patch_offset']
        refs[row['entry_offset']:row['entry_offset']+row['samples']] = inverse + offset
        canvas = NativeCanvas(row)
        for start in range(0, len(coords), 32):
            parts = [canvas.crop(int(yy)-112, int(xx)-112, 352) for yy, xx in coords[start:start+32]]
            a = np.stack([p[0] for p in parts])
            b = np.stack([p[1] for p in parts])
            x[offset+start:offset+start+len(parts)].copy_(torch.from_numpy(a))
            y[offset+start:offset+start+len(parts)].copy_(torch.from_numpy(b))
        del canvas
        print('PACKED_TRAIN', row['id'], len(coords), device, flush=True)
    host = dict(x=x, y=y) if keep_cpu else None
    if keep_cpu and required + 36 * 2**30 >= torch.cuda.mem_get_info()[0]:
        device = 'cpu'
    print('CONTEXT_ARENA', dict(cpu_backing=keep_cpu, device=device,
                              free_gpu_bytes=torch.cuda.mem_get_info()[0]), flush=True)
    if keep_cpu and device == 'cuda':
        x, y = x.cuda(), y.cuda()
    return dict(x=x, y=y, host=host, refs=refs, doc=doc,
                augmentation_rng=np.random.default_rng(2026090719),
                packing=dict(device=device, native_uint8_bytes=required, measured_parent=352,
                             layout='NHWD; depth contiguous', distinct_patches=count))


def batch(data, entries, rng):
    n = len(entries)
    ids = torch.as_tensor(data['refs'][entries], device=data['x'].device)
    # Equivalent to the study's480px parent offset112 and224px fetch:
    # removing64 unused pixels per side changes the fetch origin to zero.
    offsets = rng.integers(0, 129, size=(n, 2))
    device = data['x'].device
    yy = torch.as_tensor(offsets[:, 0], device=device)[:, None, None] + torch.arange(224, device=device)[None, :, None]
    xx = torch.as_tensor(offsets[:, 1], device=device)[:, None, None] + torch.arange(224, device=device)[None, None, :]
    raw = data['x'][ids[:, None, None], yy, xx].to('cuda').permute(0, 3, 1, 2)[:, None].float()
    target = data['y'][ids[:, None, None], yy, xx].to('cuda')[:, None].float()/255
    gain = torch.as_tensor(rng.uniform(.8, 1.2, (n, 1, 1, 1, 1)), device='cuda', dtype=torch.float32)
    bias = torch.as_tensor(rng.uniform(-15, 15, (n, 1, 1, 1, 1)), device='cuda', dtype=torch.float32)
    raw = (raw*gain+bias).clamp_min_(0)
    rotation, flip = int(rng.integers(4)), bool(rng.integers(2))
    raw, target = raw.rot90(rotation, [-2, -1]), target.rot90(rotation, [-2, -1])
    if flip:
        raw, target = raw.flip(-1), target.flip(-1)
    if rng.random() < .2:
        z = int(rng.integers(16))
        raw[:, :, z] = raw.mean(2)
    raw, target = scroll_shift(raw, target, 128, data['augmentation_rng'], 'scroll-shift-v2')
    return raw.clamp_max_(200).mul_(1/255).contiguous(), target.contiguous()


def load_validation(doc, keep_cpu=False):
    rows = [r for r in doc['segments'] if r['role'] == 'validation']
    assert len(rows) == 1 and rows[0]['id'] == doc['validation_id'] and rows[0]['scroll'] in SCROLLS
    row = rows[0]
    coords = np.load(Path(row['context_folder']) / 'coordinates.npy')
    canvas = NativeCanvas(row)
    pairs = [canvas.crop(int(y), int(x), 128) for y, x in coords]
    host_x = torch.as_tensor(np.stack([p[0].transpose(2, 0, 1) for p in pairs]))
    return dict(x=host_x.cuda(), host_x=host_x if keep_cpu else None,
                y=np.stack([p[1] for p in pairs]), coords=coords, window=128, metric_core=32,
                segment=row['id'])
