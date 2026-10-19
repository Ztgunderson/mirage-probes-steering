#!/usr/bin/env bash
# slake extract (running) -> [GPU] steer (lean arms)  ||  [CPU] nb04 geometry ; then nb03
set -u
cd $HOME/lab/benches/mirage-med
PY=$HOME/lab/benches/mats-nanda/.venv/bin/python
JUP=$HOME/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter
note(){ echo "$(date +%T) $*" | tee -a runs/chain.log; }
nbrun(){ (cd review && timeout 7200 $JUP nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mats-bench \
  --ExecutePreprocessor.timeout=5400 "$1" > "../runs/$2" 2>&1) && note "$1 executed" || note "$1 FAILED (see runs/$2)"; }
while pgrep -f "[m]iragelab.extract --dataset slake" >/dev/null; do sleep 20; done
[ -f runs/slake/acts/meta.json ] || { note "slake acts missing — abort"; exit 1; }
note "slake extract done: $(tail -c 80 runs/extract-slake.log | tr '\r' '\n' | tail -1)"
( timeout 1800 $PY -m production.miragelab.steer --run runs/slake --limit 4 --max_new_tokens 24 > runs/steer-smoke.log 2>&1 \
    && note "steer smoke ok" \
    && timeout 14400 $PY -m production.miragelab.steer --run runs/slake > runs/steer.log 2>&1 \
    && note "steer done" || note "steer FAILED (see runs/steer-smoke.log / runs/steer.log)" ) &
nbrun 04-slake-behaviour-geometry.ipynb nb4.log
wait
nbrun 03-slake-embedded-and-steerable.ipynb nb3.log
note "chain-geom finished"
