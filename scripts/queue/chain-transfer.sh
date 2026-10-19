#!/usr/bin/env bash
# Queued 09-30 behind chain-dose.sh: Qwen3-VL-2B-Instruct -> Cosmos-Reason2-2B mirage-vector transfer.
# Waits until chain-dose finishes or fails (one model on the GPU at a time: NVML OOM), then
# smoke (4 texts per split) -> full: gen 2B, gen Cosmos, steer. Progress in runs/chain.log.
set -uo pipefail
cd $HOME/lab/benches/mirage-med
PY=$HOME/lab/benches/mats-nanda/.venv/bin/python
note(){ echo "$(date '+%m-%d %T') $*" | tee -a runs/chain.log; }
step(){ # name timeout args...
  local name=$1 to=$2; shift 2
  note "transfer $name start"
  timeout "$to" $PY -m production.miragelab.transfer "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Fetching|NvMap" >> "runs/transfer-$name.log" \
    && note "transfer $name ok: $(grep -E 'texts|AUC|CHOSEN' runs/transfer-$name.log | tail -1)" || { note "transfer $name FAILED (runs/transfer-$name.log)"; exit 2; }
}
note "chain-transfer queued, waiting for chain-dose"
while pgrep -f "[c]hain-dose.sh" >/dev/null; do sleep 60; done
step smoke-gen-2b     1h --stage gen --model instruct2b --smoke 4
step smoke-gen-cosmos 1h --stage gen --model cosmos2b --smoke 4
step smoke-steer      2h --stage steer --smoke 4
step gen-2b           3h --stage gen --model instruct2b
step gen-cosmos       5h --stage gen --model cosmos2b
step steer           10h --stage steer
note "chain-transfer finished"
