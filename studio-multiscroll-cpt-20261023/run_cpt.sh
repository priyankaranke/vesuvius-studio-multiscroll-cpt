#!/bin/bash
# Keeps train_cpt.py running until training-result.json says complete. Exit 75 = clean stop on SIGTERM (e.g. a manual
# revision round on the GH200 took the shared GPU lease): wait, re-acquire the lease and resume from checkpoints/resume.pt.
cd "$(dirname "$0")"
PY=${PY:-python}
failures=0
while true; do
  if [ -f training-result.json ] && grep -q '"status": "complete"' training-result.json; then echo "$(date -u +%FT%TZ) COMPLETE"; exit 0; fi
  [ -f STOP ] && { echo "$(date -u +%FT%TZ) STOP file present"; exit 0; }
  echo "$(date -u +%FT%TZ) START train_cpt.py"
  "$PY" -u train_cpt.py; code=$?
  echo "$(date -u +%FT%TZ) EXIT $code"
  if [ $code -eq 0 ]; then continue; fi
  if [ $code -eq 75 ]; then sleep 30; continue; fi
  failures=$((failures+1)); [ $failures -ge 6 ] && { echo "giving up after $failures failures"; exit 1; }
  sleep 120
done
