#!/usr/bin/env bash
# Run the published inference implementation with verified released weights.
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
case "${1:-}" in
  -h|--help) echo 'Usage: bash infer.sh --input planes16.npy --output ink.npy [--device cpu] [--checkpoint PATH]'; exit 0 ;;
esac
if [ ! -x "$HERE/.venv-inference/bin/python" ]; then
  uv venv --python 3.11.16 "$HERE/.venv-inference"
fi
uv pip install --python "$HERE/.venv-inference/bin/python" --index-strategy unsafe-best-match -r "$HERE/requirements-inference.txt"
exec "$HERE/.venv-inference/bin/python" "$HERE/infer_release.py" "$@"
