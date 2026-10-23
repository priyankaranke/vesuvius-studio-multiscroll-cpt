"""Studio after multi-scroll continued pretraining ('studioms') on a frozen native-17 manual revision: the same input recipe and
engine as the dashboard Studio (requested_models.predict with the reference engine, which takes an explicit checkpoint hash);
only the checkpoint differs. Output name 'studioms'. Copied into a manual family folder next to its infer.py.
usage: infer_studioms.py <revision or history/original dir> <model.safetensors> [--name studioms]"""
import os
os.environ.update(OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', OMP_NUM_THREADS='16', TORCHINDUCTOR_COMPILE_THREADS='4')
import gc, hashlib, json, sys, time
from pathlib import Path
# The optimized engine pins the stock Studio hash; the reference engine takes an explicit hash.
sys.path.insert(0, '/home/ubuntu/PHerc0175A-fullheight-20260927/source/control-inference')
sys.path.append('/home/ubuntu/first-letters-hybrid/core')   # extra_inference, imported by requested_models.predict
CKPT = Path(sys.argv[2]).resolve(); NAME = sys.argv[sys.argv.index('--name') + 1] if '--name' in sys.argv else 'studioms'
sys.argv = sys.argv[:2]
src = Path(__file__).with_name('infer.py').read_text().split('\nwith acquire():')[0]   # the family infer.py's setup + commit(), not its model loop
ns = {'__file__': str(Path(__file__).with_name('infer.py')), '__name__': 'family_infer_commit'}; exec(compile(src, 'infer.py', 'exec'), ns)
out, raw, sign, voxel_um, commit = ns['out'], ns['raw'], ns['sign'], ns['voxel_um'], ns['commit']
from first_letters_inference.lease import acquire
from first_letters_inference.engine import Engine
from requested_models import predict
sha = hashlib.sha256(CKPT.read_bytes()).hexdigest()
with acquire():
    import torch
    if not (out/(NAME+'.npy')).exists():
        tick = time.monotonic(); print('MODEL_START', out.name, NAME, flush=True)
        engine = Engine('studio', CKPT, checkpoint_sha256=sha)
        p, cov, stats = predict(engine, 'studio', out, raw, sign, voxel_um, out, lambda done, total: None)
        commit(NAME, p, cov, dict(stats, checkpoint=str(CKPT), trained_on='Studio epoch 9, continued on 12 other scroll groups (Expanded9 corpus + Reader v2/v3 labels) with 814/841/9B replay; '
                                   'HF paranke/vesuvius-studio-multiscroll-cpt-20261023'), time.monotonic()-tick, sha)
        del engine, p, cov; gc.collect(); torch.cuda.empty_cache()
print('STUDIOMS_DONE', out.name, NAME, flush=True)
