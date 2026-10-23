#!/usr/bin/env bash
# Run the multi-scroll Studio ('studioms') on the PHerc0211 crop 1711 DinoVol revision family (Original + revision 1) with the
# dashboard Studio recipe, add it to the 8798 viewer as layer 8 and register the Original's result into the revision.
# usage: publish_crop1711.sh <model.safetensors>
set -euo pipefail
CKPT=$(readlink -f "$1")
BASE=/home/ubuntu/first-letters-will-20261007/manual/PHerc0211-crop1711-dinogrow-20261023
V=/home/ubuntu/first-letters-will-20261007/viewer/dist/PHerc0211-crop1711-dinogrow-revisions
ORIG=$V/history/original; OUT=$V/revisions/PHerc0211-crop1711-dinogrow-20261023-legacyuv20-v1
CPU=/home/ubuntu/PHerc0846A-spiral/env/fit/bin/python; GPU=$CPU
INF=/home/ubuntu/first-letters-inference-optimized:/home/ubuntu/PHerc0175A-fullheight-20260927/runtime/ensemble-ink:/home/ubuntu/PHerc0175A-fullheight-20260927/runtime/cudnn-frontend
HERE=$(cd "$(dirname "$0")" && pwd)
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONUNBUFFERED=1
cp "$HERE/infer_studioms.py" "$HERE/add_studioms_layer.py" "$BASE/"
T0=$(date +%s); el() { echo "$1 $(( $(date +%s)-T0 ))s"; }
for S in "$ORIG" "$OUT"; do PYTHONPATH=$INF:/home/ubuntu/pherc1545-willx-20261020:$BASE "$GPU" "$BASE/infer_studioms.py" "$S" "$CKPT"; done; el INFER_DONE
"$CPU" "$BASE/add_studioms_layer.py" "$BASE" "$V"
"$CPU" "$BASE/register3d.py" "$OUT"; el REGISTER_DONE
"$CPU" "$BASE/build_overlays.py" "$OUT"
"$CPU" "$BASE/make_jpegs.py" "$V"; el PAGE_READY
