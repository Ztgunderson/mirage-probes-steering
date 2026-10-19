#!/usr/bin/env bash
# mcq gen (running) -> prior -> subset -> stop vLLM -> extract -> execute notebook 02
set -u
cd $HOME/lab/benches/mirage-med
PY=$HOME/lab/benches/mats-nanda/.venv/bin/python
note(){ echo "$(date +%T) $*" | tee -a runs/chain.log; }
while pgrep -f "gen_mutations_get_responses.py.*runs/mcq400" >/dev/null; do sleep 60; done
[ -f runs/mcq400/responses.json ] || { note "mcq gen produced no responses.json — abort"; exit 1; }
note "mcq gen done"
timeout 1800 $PY -m production.miragelab.prior --run runs/mcq400 --workers 16 >> runs/chain.log 2>&1 || { note "mcq prior FAILED"; exit 1; }
note "mcq subset: $($PY -c "from production.miragelab.analysis import subset; print(subset('runs/mcq400'))")"
timeout 90 docker stop mirage-qwen3vl-4b >/dev/null 2>&1; sleep 10
note "mcq extract start"
timeout 14400 $PY -m production.miragelab.extract --dataset medxpertqa_mm --max_pixels 1003520 \
  --responses runs/mcq400/responses_subset.json --out runs/mcq400/acts > runs/extract-mcq.log 2>&1 || { note "mcq extract FAILED (see runs/extract-mcq.log)"; exit 1; }
note "mcq extract done: $(tail -1 runs/extract-mcq.log)"
cd review && timeout 7200 $HOME/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=mats-bench --ExecutePreprocessor.timeout=5400 02-yesno-vs-mc-prior-controlled.ipynb > ../runs/nb2.log 2>&1 \
  && note "notebook 02 executed" || note "notebook 02 FAILED (see runs/nb2.log)"
