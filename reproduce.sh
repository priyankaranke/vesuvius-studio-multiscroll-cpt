#!/usr/bin/env bash
# Reproduce the Studio soft-loss ink model (the weights behind the 28 Oct 2026 PHerc0211 submission) from shared data.
#
#   bash reproduce.sh            full chain: Studio data + other-scroll data, continued pretraining (30,000 updates, ~9 h),
#                                soft-loss stage (3,000 updates, ~1 h); one 96 GB GPU (GH200/H100), ~120 GB of downloads
#   bash reproduce.sh softloss   only the soft-loss stage (~1 h), starting from the released continued-pretraining weights
#
# Needs: Python 3.11 with uv (or make .venv yourself from requirements-training.txt), the GitHub CLI logged in to an account
# with access to this repository (release assets), and Hugging Face read access to the private datasets (hf auth login).
# The PHerc0211 held-out evaluation set is hand-labelled review material and is not public: without it the training
# scripts skip that evaluation and train identically; set PHERC0211_EVAL_DATA=<dir> to enable it.
# Layout created next to this checkout: ../expanded9, ../reader-v2, ../reader-rows, ../weights (what the scripts expect).
set -euo pipefail
HERE=$(cd "$(dirname "$0")" && pwd); WORK=$(dirname "$HERE"); STEP=${1:-all}; cd "$HERE"
if [ ! -x .venv/bin/python ]; then
  uv venv --python 3.11.16 .venv
  uv pip install --python .venv --index-strategy unsafe-best-match -r requirements-training.txt
fi
PY=$HERE/.venv/bin/python; HF=$HERE/.venv/bin/hf
export STUDIO_GPU_LEASE=0 CPT_DROP_CACHES=${CPT_DROP_CACHES:-0}

echo "== 1/4 Studio training data (tiles, labels, plans) and the Studio starting weights"
"$PY" download_data.py
mkdir -p "$WORK/weights/vesuvius-814-841-9b-resunet-studio-20260926"
"$HF" download paranke/vesuvius-814-841-9b-resunet-studio-20260926 model.safetensors --local-dir "$WORK/weights/vesuvius-814-841-9b-resunet-studio-20260926"

echo "== 2/4 other-scroll rows: Expanded9 corpus and Reader v2/v3 labels"
"$HF" download paranke/vesuvius-expanded9-training-data --repo-type dataset --include 'prepared/*' --local-dir "$WORK/expanded9"
mkdir -p "$WORK/reader-v2"
"$HF" download domenicor046/reader-v2-labels --repo-type dataset --local-dir "$WORK/reader-v2"
[ -d "$WORK/reader-v2/reader-v2-labels" ] || tar -xf "$WORK/reader-v2/reader-v2-dense-native-labels.tar" -C "$WORK/reader-v2"
[ -d "$WORK/reader-rows" ] || "$PY" studio-multiscroll-cpt-20261023/prepare_reader.py --labels "$WORK/reader-v2/reader-v2-labels" \
    --manifest "$WORK/reader-v2/manifest.csv" --out "$WORK/reader-rows" --objects 0139 0500P2 0343P 0172 --max-patches 0172=700

echo "== 3/4 continued pretraining (studio-multiscroll-cpt-20261023)"
if [ "$STEP" = all ]; then
  (cd studio-multiscroll-cpt-20261023 && PY="$PY" bash run_cpt.sh)
else
  mkdir -p studio-multiscroll-cpt-20261023/checkpoints
  gh release download studio-multiscroll-cpt-20261023 -R priyankaranke/vesuvius-studio-multiscroll-cpt \
      -p 'studio-multiscroll-cpt-20261023-ema.safetensors' -O studio-multiscroll-cpt-20261023/checkpoints/model-ema.safetensors --clobber
fi

echo "== 4/4 soft-loss stage (studio-softloss-20261023)"
(cd studio-softloss-20261023 && PY="$PY" bash run_softloss.sh)
echo "DONE: $HERE/studio-softloss-20261023/checkpoints/model-ema.safetensors"
