#!/usr/bin/env bash
# Full training from random initialization, using all three linked datasets.
# bash reproduce.sh          scratch (9 epochs) -> other-scroll training -> soft-loss
# bash reproduce.sh softloss final stage only, from the released intermediate weights
# Use a fresh clone. Re-running the same command resumes that reproduction.
# Needs uv, Hugging Face dataset access, one NVIDIA GPU with >=96 GB VRAM,
# >=128 GB host RAM and >=200 GB free disk. The full run takes longer than 10 hours.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
cd "$HERE"
case "${1:-all}" in
  all|softloss) ;;
  -h|--help) echo 'Usage: bash reproduce.sh [all|softloss]'; exit 0 ;;
  *) echo 'Usage: bash reproduce.sh [all|softloss]' >&2; exit 2 ;;
esac
if [ "$#" -gt 1 ]; then echo 'Only one mode argument is accepted' >&2; exit 2; fi
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11.16 .venv
fi
uv pip install --python .venv --index-strategy unsafe-best-match -r requirements-training.txt
exec "$HERE/.venv/bin/python" "$HERE/reproduction.py" "${1:-all}"
