#!/usr/bin/env bash
# slake+mcq gen (running) -> mcq prior -> stop vLLM -> extract slake -> pick layer (train/val) -> steer (test)
#   -> execute nb03 -> extract mcq subset -> execute nb02
set -u
cd $HOME/lab/benches/mirage-med
PY=$HOME/lab/benches/mats-nanda/.venv/bin/python
JUP=$HOME/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter
note(){ echo "$(date +%T) $*" | tee -a runs/chain.log; }
nbrun(){ (cd review && timeout 7200 $JUP nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mats-bench \
  --ExecutePreprocessor.timeout=5400 "$1" > "../runs/$2" 2>&1) && note "$1 executed" || note "$1 FAILED (see runs/$2)"; }
while pgrep -f "[g]en_mutations_get_responses.py.*runs/mcq400" >/dev/null || pgrep -f "[m]iragelab.slake " >/dev/null; do sleep 60; done
note "gens done: slake=$(tail -1 runs/slake.log)"
[ -f runs/mcq400/responses.json ] && timeout 1800 $PY -m production.miragelab.prior --run runs/mcq400 --workers 16 >> runs/chain.log 2>&1
timeout 90 docker stop mirage-qwen3vl-4b >/dev/null 2>&1; sleep 10
[ -f runs/slake/responses.json ] || { note "no slake responses — abort"; exit 1; }
note "slake extract start"
timeout 10800 $PY -m production.miragelab.extract --dataset slake --responses runs/slake/responses.json --out runs/slake/acts \
  > runs/extract-slake.log 2>&1 || { note "slake extract FAILED"; exit 1; }
LAYER=$($PY -c "
from production.miragelab import analysis as A
df, acts, meta = A.load('runs/slake'); y = df.outcome.isin(['hedged','mirage']).values
j, aucs = A.best_layer(acts, y, (df.split=='train').values, (df.split=='validation').values)
print(meta['layers'][j])")
note "slake extract done; steering layer (train fit, val AUC) = L$LAYER"
timeout 1800 $PY -m production.miragelab.steer --run runs/slake --layer "$LAYER" --limit 4 --max_new_tokens 24 > runs/steer-smoke.log 2>&1 \
  && note "steer smoke ok" || { note "steer smoke FAILED (see runs/steer-smoke.log)"; exit 1; }
timeout 14400 $PY -m production.miragelab.steer --run runs/slake --layer "$LAYER" > runs/steer.log 2>&1 && note "steer done: $(grep -c image runs/steer.log) arms" || note "steer FAILED (see runs/steer.log)"
nbrun 03-slake-embedded-and-steerable.ipynb nb3.log
if [ -f runs/mcq400/responses.json ]; then
  note "mcq subset: $($PY -c "from production.miragelab.analysis import subset; print(subset('runs/mcq400'))")"
  timeout 14400 $PY -m production.miragelab.extract --dataset medxpertqa_mm --max_pixels 1003520 \
    --responses runs/mcq400/responses_subset.json --out runs/mcq400/acts > runs/extract-mcq.log 2>&1 && nbrun 02-yesno-vs-mc-prior-controlled.ipynb nb2.log || note "mcq extract FAILED"
fi
note "chain-slake finished"
