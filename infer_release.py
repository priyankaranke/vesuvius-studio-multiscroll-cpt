"""Download/verify the submission weights and invoke the in-repository inference CLI."""
import argparse
from pathlib import Path
import subprocess
import sys

from reproduction import sha256

WEIGHT_SHA256 = '96b326286e38de99047f7cab36170c53749a18d5f43e5fde0c7c20f2980b0e90'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='uint8 .npy [16,H,W], planes 23 down to 8')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--checkpoint', type=Path, help='Existing weights; defaults to the released soft-loss weights')
    parser.add_argument('--checkpoint-sha256', default=WEIGHT_SHA256,
                        help='Expected hash; set explicitly for newly trained weights')
    parser.add_argument('--device', choices=['cpu', 'cuda', 'mps'], default='cuda')
    parser.add_argument('--batch-size', type=int, help='Defaults to 1 on Apple GPUs and 32 elsewhere')
    parser.add_argument('--amp', choices=['bf16', 'fp32'], default='bf16')
    parser.add_argument('--stride', type=int, default=64,
                        help='Window step with Hann blending; 64 = the submission tiling, 48 = no tile pattern (~1.7x slower)')
    args = parser.parse_args(argv)
    if not args.input.is_file():
        parser.error('Input does not exist; render and stack the 16 CT planes first')
    if args.output.exists():
        parser.error('Output already exists; choose a new path')
    root = Path(__file__).resolve().parent
    checkpoint = args.checkpoint
    if checkpoint is None:
        checkpoint = root / '.inference/model-ema.safetensors'
        if not checkpoint.exists():
            checkpoint.parent.mkdir(exist_ok=True)
            partial = checkpoint.with_suffix('.partial')
            subprocess.run(['gh', 'release', 'download', 'studio-softloss-20261023', '-R',
                            'priyankaranke/vesuvius-studio-multiscroll-cpt', '-p',
                            'studio-softloss-20261023-ema.safetensors', '-O', str(partial), '--clobber'], check=True)
            if sha256(partial) != WEIGHT_SHA256:
                raise RuntimeError('Downloaded model checksum does not match the released submission weights')
            partial.replace(checkpoint)
    if sha256(checkpoint) != args.checkpoint_sha256:
        raise RuntimeError('Model checksum mismatch; no inference was run')
    batch_size = args.batch_size
    if batch_size is None:
        batch_size = 1 if args.device == 'mps' else 32
    subprocess.run([sys.executable, str(root / 'studio-multiscroll-cpt-20261023/hf/inference.py'),
                    '--input', str(args.input.resolve()), '--output', str(args.output.resolve()),
                    '--checkpoint', str(checkpoint.resolve()), '--stride', str(args.stride), '--blend', 'hann',
                    '--batch-size', str(batch_size), '--device', args.device, '--amp', args.amp], check=True)


if __name__ == '__main__':
    main()
