#!/usr/bin/env bash
# CPU: wait nb04 -> nb05 -> nb01 -> nb02 ; then wait steering -> nb03
set -u
cd $HOME/lab/benches/mirage-med
JUP=$HOME/lab/existing/jetson-llm/stance-drift/.venv/bin/jupyter
note(){ echo "$(date +%T) $*" | tee -a runs/chain.log; }
nbrun(){ (cd review && timeout 7200 $JUP nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name=mats-bench \
  --ExecutePreprocessor.timeout=5400 "$1" > "../runs/$2" 2>&1) && note "$1 executed" || note "$1 FAILED (see runs/$2)"; }
while pgrep -f "[n]bconvert.*04-slake" >/dev/null; do sleep 20; done
nbrun 05-pressure-vector.ipynb nb5.log
nbrun 01-mirage-4b-replication.ipynb nb.log
nbrun 02-yesno-vs-mc-prior-controlled.ipynb nb2.log
while pgrep -f "[m]iragelab.steer --run runs/slake" >/dev/null || ! grep -q "steer done (held-out\|steer FAILED" runs/chain.log; do sleep 30; done
nbrun 03-slake-behaviour-and-steering.ipynb nb3.log
note "chain-rerun finished"
