#!/usr/bin/env bash
# 10-01: dose.py on split seeds 1-4 (seed 0 = runs/slake/dose, done 09-30) -> seed-0 T=0.7 samples (from the
# stuck chain-samples.sh) -> pool. Same arms/config as seed 0. A failed seed is logged and skipped, not fatal.
# Resume = rerun this script: dose/test skip arms already in their rows files (fit reruns, ~6 min).
set -uo pipefail
cd $HOME/lab/benches/mirage-med
PY=$HOME/lab/benches/mats-nanda/.venv/bin/python
note(){ echo "$(date '+%m-%d %T') $*" | tee -a runs/chain.log; }
stage(){ # seed name timeout args...
  local s=$1 name=$2 to=$3; shift 3
  note "folds s$s $name start"
  timeout "$to" $PY -m production.miragelab.dose --run runs/slake --seed "$s" "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Fetching|NvMap" >> "runs/folds-s$s-$name.log" \
    && note "folds s$s $name ok: $(grep -E 'GATE|CHOSEN' runs/folds-s$s-$name.log | tail -1)" || { note "folds s$s $name FAILED (runs/folds-s$s-$name.log)"; return 1; }
}
for s in 1 2 3 4; do
  stage $s fit 1h --stage fit && stage $s dose 6h --stage dose && stage $s test 12h --stage test || note "folds s$s abandoned"
done
ARMS="general,leftover,prompt + template direction (leftover alpha),prompt + leftover minus template (leftover alpha),prompt + shuffled-label (leftover alpha),prompt + random (leftover alpha)"
stage 0 samples 6h --stage test --samples 3 --sample_arms "$ARMS"
$PY -m production.miragelab.dose --run runs/slake --stage pool > runs/folds-pool.txt 2>&1 && note "folds pooled: runs/folds-pool.txt"
note "chain-folds finished"
