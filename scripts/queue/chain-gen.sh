#!/usr/bin/env bash
# 10-01 evening, replaces chain-next.sh (same steps, reordered; folds s3/s4 dropped for now).
# Finishes the folds s2 test (resume), then (one model on the GPU at a time: NVML OOM):
# smoke block (GATE: any failure -> "SMOKE FAILED: <step>", exit) -> sweep-s0 (direction vs size)
# -> generalisation: 2B/Cosmos transfer, prompts on 2B/Cosmos, 4B-Thinking, 4B vectors on folds 1-4
# -> introspect -> sweep-s1 -> pool. Each step timeout-wrapped, logged to runs/chain.log (+ runs/next-<step>.log).
set -uo pipefail
cd $HOME/lab/benches/mirage-med
PY=$HOME/lab/benches/mats-nanda/.venv/bin/python
OK=runs/.think4b-downloaded
note(){ echo "$(date '+%m-%d %T') $*" | tee -a runs/chain.log; }
step(){ # name timeout module args...
  local name=$1 to=$2 mod=$3; shift 3
  note "next $name start"
  timeout --kill-after=5m "$to" $PY -m "production.miragelab.$mod" "$@" 2>&1 | grep --line-buffered -vE "Loading weights|Fetching|NvMap" > "runs/next-$name.log"
  local rc=${PIPESTATUS[0]}
  if [ "$rc" = 0 ]; then note "next $name ok: $(grep -E 'texts|CHOSEN|AUC|sweep|parsed|seeds|mirage' "runs/next-$name.log" | tail -1 | cut -c1-160)"
  else note "next $name FAILED rc=$rc (runs/next-$name.log)"; fi
  return "$rc"
}

note "chain-gen start (folds s2 test resumes arm by arm; its 16:56 run was killed mid-arm)"
step folds-s2-test 8h dose --run runs/slake --seed 2 --stage test

# ---- smoke block (GPU) ----
rm -rf runs/slake/dose_smoke_s1 runs/slake/introspect_smoke runs/transfer_smoke runs/slake/dose_sweep_smoke \
       runs/slake/dose_think4b_smoke runs/slake/dose_instruct2b_smoke runs/slake/dose_cosmos2b_smoke
fail(){ note "SMOKE FAILED: $1"; exit 1; }
step smoke-xfer-s1    1h dose --run runs/slake --stage test --seed 1 --vectors_from 0 --smoke 4 || fail xfer-s1
step smoke-introspect 1h introspect --run runs/slake --smoke 4 || fail introspect
step smoke-gen-2b     1h transfer --stage gen --model instruct2b --smoke 8 || fail gen-2b
step smoke-gen-cosmos 1h transfer --stage gen --model cosmos2b --smoke 8 || fail gen-cosmos
step smoke-steer      2h transfer --stage steer --smoke 8 || fail transfer-steer
step smoke-prompts-2b     1h dose --run runs/slake --stage test --model instruct2b --prompts_only --smoke 4 || fail prompts-2b
step smoke-prompts-cosmos 1h dose --run runs/slake --stage test --model cosmos2b --prompts_only --smoke 4 --max_new_tokens 512 || fail prompts-cosmos
if [ -f $OK ]; then
  step smoke-think4b  1h dose --run runs/slake --stage test --model think4b --smoke 4 --max_new_tokens 512 || fail think4b
else note "next smoke-think4b skipped: model not downloaded"; fi
step smoke-sweep      1h dose --run runs/slake --stage sweep --seed 0 --smoke 4 || fail sweep
note "SMOKE OK: all smoke steps passed, starting full runs"

# ---- full runs ----
step sweep-s0 18h dose --run runs/slake --stage sweep --seed 0
step gen-2b 3h transfer --stage gen --model instruct2b \
  && step gen-cosmos 5h transfer --stage gen --model cosmos2b \
  && step transfer-steer 10h transfer --stage steer
step prompts-2b     3h dose --run runs/slake --stage test --model instruct2b --prompts_only --test_image 60
step prompts-cosmos 12h dose --run runs/slake --stage test --model cosmos2b --prompts_only --test_image 60
if [ -f $OK ]; then step think4b 24h dose --run runs/slake --stage test --model think4b --test_image 60
else note "next think4b skipped: model not downloaded"; fi
for s in 1 2 3 4; do step xfer-s$s 8h dose --run runs/slake --stage test --seed $s --vectors_from 0; done
step introspect 3h introspect --run runs/slake
step sweep-s1 18h dose --run runs/slake --stage sweep --seed 1
step pool 30m dose --run runs/slake --stage pool && cp runs/next-pool.log runs/next-pool.txt
note "chain-gen finished"
