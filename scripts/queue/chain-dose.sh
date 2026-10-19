#!/usr/bin/env bash
# Overnight 09-30: finish the dose.py smoke (3 control arms) -> full fit -> dose -> test (+3 samples).
# Detached (setsid nohup); every stage timeout-wrapped; progress in runs/chain.log, output in runs/dose-*.log.
# Resume = rerun this script: dose/test skip arms already in their rows file (fit reruns, ~15 min).
set -uo pipefail
cd $HOME/lab/benches/mirage-med
PY=$HOME/lab/benches/mats-nanda/.venv/bin/python
note(){ echo "$(date '+%m-%d %T') $*" | tee -a runs/chain.log; }
stage(){ # name timeout args...
  local name=$1 to=$2; shift 2
  note "dose $name start"
  timeout "$to" $PY -m production.miragelab.dose --run runs/slake "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Fetching|NvMap" >> "runs/dose-$name.log" \
    && note "dose $name ok: $(grep -E 'GATE|CHOSEN' runs/dose-$name.log | tail -1)" || { note "dose $name FAILED (runs/dose-$name.log)"; exit 2; }
}
stage smoke-test 1h --stage test --smoke 12 --max_new_tokens 256
stage fit        1h --stage fit
stage dose       6h --stage dose
stage test      12h --stage test --samples 3
note "chain-dose finished"
