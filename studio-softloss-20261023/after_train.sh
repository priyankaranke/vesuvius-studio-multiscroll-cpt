#!/bin/bash
# Step 2 evaluation after training: soft-loss EMA maps at stride 96 (AP, same tiling as step 1) and 48 (display), then compare.
set -e
S=/home/ubuntu/studio-multiscroll-cpt-20261023/studio/studio-softloss-20261023; PY=/home/ubuntu/donor-continued-training-20260910/.venv/bin/python
cd $S
CK=$S/checkpoints/model-ema.safetensors
$PY -u soft_check.py infer --checkpoint $CK --name softloss
$PY -u soft_check.py infer --checkpoint $CK --name softloss-s48 --stride 48
$PY -u soft_check.py analyze --models studioms,softloss --tag=-s96
$PY -u soft_check.py analyze --models studioms-s48,softloss-s48 --tag=-s48
echo AFTER_TRAIN_DONE
