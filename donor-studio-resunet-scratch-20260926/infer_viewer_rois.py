"""Epoch 9 Studio checkpoint on the requested native viewer ROIs."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('TORCHINDUCTOR_COMPILE_THREADS', '16')
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parent
FAMILY = 'donor-studio-sparse-epoch9-s64-hann'
WEIGHTS_SHA = 'ffffbf4fcb2f1eb91940a6899366cd0d22b2ea03987270d07dc2a205dbd91145'
OUTPUT = ROOT / 'roi-inference-epoch9'
FAMILY_ROOT = OUTPUT / 'viewer-family' / FAMILY
PACKAGE = ROOT.parent / 'pherc1447-firstletters-i3d-roi-20260915/label-source-20260916-r00/segments'
EXPECTED = {
    'w056-crop-01': [1710, 11865, 6000, 6000],
    'w056-crop-02': [3060, 15672, 6000, 6000],
    'w056-crop-03': [2113, 2786, 6000, 6000],
    'w090-crop-01': [878, 6724, 6000, 6000],
    'w090-crop-02': [183, 724, 6000, 6000],
    'w090-crop-03': [6183, 724, 6000, 6000],
    'w097-crop-01': [8020, 11080, 6000, 6000],
    'w097-crop-02': [1071, 2511, 6000, 6000],
    'w097-crop-03': [7071, 4608, 6000, 6000],
    'w066-crop-01': [3847, 9125, 6000, 6000],
    'w066-crop-02': [3533, 100, 6000, 6000],
}


def atomic(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(value, indent=2) + '\n')
    tmp.replace(path)


def sha(path):
    result = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            result.update(block)
    return result.hexdigest()


def check_lease():
    token = os.environ['DONOR_SPIRAL_GPU_LEASE_TOKEN']
    value = json.loads((ROOT.parent / '.cache/vesuvius-gpu/donor-priority.json').read_text())
    assert value['status'] == 'running' and value['ownership_token'] == token


def load_input(rid):
    import numpy as np
    import tifffile
    from PIL import Image
    folder = ROOT / 'roi-inputs' / rid
    deadline = time.monotonic() + 3600
    metadata = folder / 'PROVENANCE.json'
    while True:
        check_lease()
        try:
            source = json.loads(metadata.read_text())
            break
        except FileNotFoundError:
            pass
        except json.JSONDecodeError:
            # The uploader copies this last; it can briefly be empty in flight.
            if time.time() - metadata.stat().st_mtime > 30:
                raise
        if time.monotonic() > deadline:
            raise RuntimeError('Timed out waiting for requested input ' + rid)
        time.sleep(1)
    assert source['id'] == rid and source['crop_xywh'] == EXPECTED[rid]
    assert source['layer_mapping'] == 'dashboard 00..15 = source expected layers 07..22'
    assert source['native_xy_preserved'] and source['resampling'] == 'none'
    raw = np.empty((16, 6000, 6000), np.uint8)
    layers = {r['name']: r for r in source['layers']}
    for z in range(16):
        name = f'{z:02d}.tif'
        path = folder / 'layers' / name
        assert sha(path) == layers[name]['sha256'], (rid, name, 'source checksum differs')
        values = tifffile.imread(path, maxworkers=1)
        assert values.shape == (6000, 6000) and values.dtype == np.uint8
        raw[z] = values
    mask = PACKAGE / rid / source['physical_mask']['filename']
    assert sha(mask) == source['physical_mask']['sha256']
    support = np.asarray(Image.open(mask)) > 0
    assert support.shape == (6000, 6000) and int(support.sum()) == source['physical_mask']['valid_pixels']
    print('INPUT_LOADED', rid, flush=True)
    return raw, support, source


def main():
    check_lease()
    import numpy as np
    import torch
    from PIL import Image
    from safetensors.torch import load_file
    sys.path[:0] = [str(ROOT.parent / 'pherc1447-architecture-study-20260906'),
                    str(ROOT.parent / 'donor-original-recipe-20260910')]
    from small_unets import DeepResidualUNet
    from full_winding_inference import infer_winding
    config = json.loads((ROOT / 'hf-epoch-9/model-config.json').read_text())
    assert config['completed_epochs'] == 9 and config['updates'] == 74250
    weights = ROOT / 'hf-epoch-9/model.safetensors'
    assert sha(weights) == WEIGHTS_SHA == config['model_sha256']
    torch.set_num_threads(16)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    model = DeepResidualUNet(deeper=False)
    model.load_state_dict(load_file(str(weights)), strict=True)
    model.cuda().eval()
    torch.nn.utils.convert_conv3d_weight_memory_format(model, torch.channels_last_3d)
    runtime = torch.compile(model, dynamic=False)
    family = dict(schema_version=1, id=FAMILY,
        label='814/841/9B · Studio sparse labels · epoch 9',
        model_id=config['run_id'], source_round_id=ROOT.name, palette='Google Turbo',
        probability_encoding='uint16 ink probability /65535; fixed 0–1 scale; no per-image normalization',
        source_modes=dict(orientation='renderer_index_ascending', spacing_um=8.64, keybind=0),
        source_sha256=WEIGHTS_SHA, checkpoint_epoch=9, checkpoint_step=74250,
        initialization='scratch', s1_samples=0,
        inference=dict(patch=128, stride=64, core=96, views=1, rotations_degrees=[0], depth_reversed=False),
        model_url='https://huggingface.co/paranke/vesuvius-814-841-9b-resunet-studio-20260926/tree/epoch-9')
    atomic(FAMILY_ROOT / 'family.json', family)
    completed = []
    order = [f'{s}-crop-{i:02d}' for i in range(1, 4) for s in ('w056', 'w090', 'w097')]
    order.extend(['w066-crop-01', 'w066-crop-02'])
    total_rois = len(order)
    saved = OUTPUT / 'inference.json'
    if saved.exists():
        previous = json.loads(saved.read_text())
        assert previous['family_id'] == FAMILY and previous['model_sha256'] == WEIGHTS_SHA
        completed = previous['completed']
        assert [row['roi_id'] for row in completed] == order[:len(completed)]
        for row in completed:
            rid = row['roi_id']
            assert row['native_roi_xywh'] == EXPECTED[rid]
            assert row['checkpoint_epoch'] == 9 and row['checkpoint_step'] == 74250
            assert row['model_sha256'] == WEIGHTS_SHA
            winding, crop = rid.split('-crop-')
            path = FAMILY_ROOT / winding / ('crop-' + crop + '.png')
            assert sha(path) == row['viewer_png_sha256'], (rid, 'saved prediction differs')
        order = order[len(completed):]
        print('RESUMING_COMPLETED', len(completed), flush=True)
    atomic(OUTPUT / 'progress.json', dict(status='waiting_for_input', completed_rois=len(completed),
        total_rois=total_rois, updated_at=time.time()))
    atomic(OUTPUT / 'inference.json', dict(status='running', family_id=FAMILY, model_sha256=WEIGHTS_SHA,
        total_rois=total_rois, completed=completed))
    with ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(load_input, order[0]) if order else None
        for index, rid in enumerate(order):
            raw, support, source = future.result()
            if index + 1 < len(order):
                future = pool.submit(load_input, order[index + 1])
            check_lease()
            def report(done, total):
                atomic(OUTPUT / 'progress.json', dict(status='running', roi=rid, tiles_done=done,
                    tiles_total=total, completed_rois=len(completed), total_rois=total_rois, updated_at=time.time()))
            values, engine = infer_winding(runtime, raw, support, check_lease, rid, report)
            assert values.shape == (6000, 6000) and np.isfinite(values).all()
            winding, crop = rid.split('-crop-')
            target = FAMILY_ROOT / winding / ('crop-' + crop + '.png')
            target.parent.mkdir(exist_ok=True)
            temporary = target.with_suffix('.tmp')
            Image.fromarray(np.rint(values.clip(0, 1) * 65535).astype(np.uint16)).save(temporary, format='PNG', compress_level=4)
            temporary.replace(target)
            record = dict(roi_id=rid, native_roi_xywh=EXPECTED[rid], checkpoint_epoch=9,
                checkpoint_step=74250, model_sha256=WEIGHTS_SHA, viewer_png_sha256=sha(target),
                surface_pixels=int(support.sum()), source_layer_mapping=source['layer_mapping'],
                mask_sha256=source['physical_mask']['sha256'], coverage_fraction=1., engine=engine)
            atomic(target.with_suffix('.json'), record)
            completed.append(record)
            atomic(OUTPUT / 'inference.json', dict(status='running', family_id=FAMILY, model_sha256=WEIGHTS_SHA,
                total_rois=total_rois, completed=completed))
            print('ROI_COMPLETE', len(completed), total_rois, rid, json.dumps(engine), flush=True)
            del raw, support, values
    atomic(OUTPUT / 'inference.json', dict(status='complete', family_id=FAMILY, model_sha256=WEIGHTS_SHA,
        total_rois=total_rois, completed=completed))
    atomic(OUTPUT / 'progress.json', dict(status='complete', completed_rois=len(completed), total_rois=total_rois, updated_at=time.time()))


if __name__ == '__main__':
    try:
        main()
    except BaseException as error:
        atomic(OUTPUT / 'progress.json', dict(status='failed', error=repr(error), updated_at=time.time()))
        raise
