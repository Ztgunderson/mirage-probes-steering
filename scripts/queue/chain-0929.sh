#!/usr/bin/env bash
# gen (running) -> stop vLLM -> extract full -> execute review notebook
set -u
cd $HOME/lab/benches/mirage-med
note(){ echo "$(date +%T) $*" | tee -a runs/chain.log; }
while pgrep -f "gen_mutations_get_responses.py.*runs/full" >/dev/null; do sleep 60; done
[ -f runs/full/responses.json ] || { note "gen produced no responses.json — abort"; exit 1; }
note "gen done: $(grep -o '"without_image_mirage_like_count": [0-9]*' runs/full/summary.json | head -1)"
timeout 90 docker stop mirage-qwen3vl-4b >/dev/null 2>&1; sleep 10
note "subset: $($HOME/lab/benches/mats-nanda/.venv/bin/python -c "from production.miragelab.analysis import subset; print(subset('runs/full'))")"
note "extract start"
timeout 18000 $HOME/lab/benches/mats-nanda/.venv/bin/python -m production.miragelab.extract \
  --responses runs/full/responses_subset.json --out runs/full/acts > runs/extract.log 2>&1 || { note "extract FAILED (see runs/extract.log)"; exit 1; }
note "extract done: $(tail -1 runs/extract.log)"
cd review && timeout 5400 $HOME/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=mats-bench --ExecutePreprocessor.timeout=3600 01-mirage-4b-replication.ipynb > ../runs/nb.log 2>&1 \
  && note "notebook executed" || note "notebook FAILED (see runs/nb.log)"
