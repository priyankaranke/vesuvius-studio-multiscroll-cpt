"""Write model.safetensors from a train.py checkpoint (checkpoints/resume.pt or last.pt): the model tensors exactly as saved,
no renaming, no dtype change. This is how the published epoch-9 weights (paranke/vesuvius-814-841-9b-resunet-studio-20260926)
relate to the run's checkpoint; `--check <published model.safetensors>` verifies tensor-for-tensor equality.
usage: export_weights.py <checkpoint.pt> <model.safetensors> [--check <reference.safetensors>]"""
import hashlib, sys
from pathlib import Path
import torch
from safetensors.torch import load_file, save_file

src, dst = Path(sys.argv[1]), Path(sys.argv[2])
saved = torch.load(src, map_location='cpu', weights_only=False, mmap=True)
state = {k: v.contiguous() for k, v in saved['state_dict'].items()}
dst.parent.mkdir(parents=True, exist_ok=True); save_file(state, str(dst))
print(f'wrote {dst} ({dst.stat().st_size} bytes, {len(state)} tensors, run {saved.get("run_id")}, step {saved.get("global_step", saved.get("step"))})')
if '--check' in sys.argv:
    ref = load_file(sys.argv[sys.argv.index('--check') + 1], device='cpu')
    same = set(ref) == set(state) and all(ref[k].dtype == state[k].dtype and torch.equal(ref[k], state[k]) for k in ref)
    print('tensors identical to the reference:', same, '| file sha256', hashlib.sha256(dst.read_bytes()).hexdigest()[:16],
          'reference', hashlib.sha256(Path(sys.argv[sys.argv.index('--check') + 1]).read_bytes()).hexdigest()[:16])
    sys.exit(0 if same else 1)
