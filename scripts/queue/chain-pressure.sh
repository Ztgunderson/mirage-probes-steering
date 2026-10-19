#!/usr/bin/env bash
# after slake extract: neutral-prompt states (GPU, alongside steering) -> nb05
set -u
cd $HOME/lab/benches/mirage-med
PY=$HOME/lab/benches/mats-nanda/.venv/bin/python
JUP=$HOME/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter
note(){ echo "$(date +%T) $*" | tee -a runs/chain.log; }
while pgrep -f "[m]iragelab.extract --dataset slake" >/dev/null || [ ! -f runs/slake/acts/meta.json ]; do sleep 20; done
timeout 3600 $PY -m production.miragelab.pressure --run runs/slake > runs/pressure.log 2>&1 \
  && note "pressure states done: $(grep -o 'min cos [0-9.]*' runs/pressure.log)" || { note "pressure FAILED (see runs/pressure.log)"; exit 1; }
(cd review && timeout 5400 $JUP nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mats-bench \
  --ExecutePreprocessor.timeout=3600 05-pressure-vector.ipynb > ../runs/nb5.log 2>&1) && note "05 executed" || note "05 FAILED (see runs/nb5.log)"
