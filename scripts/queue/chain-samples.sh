#!/usr/bin/env bash
# Queued 09-30 behind chain-transfer.sh: T=0.7 x3 no-image passes on the SLAKE test texts for the arms
# combo has to beat (template direction is tied with combo at T=0, 1 vs 1 mirage). prompt + combo [sampled] exist already.
set -uo pipefail
cd $HOME/lab/benches/mirage-med
PY=$HOME/lab/benches/mats-nanda/.venv/bin/python
note(){ echo "$(date '+%m-%d %T') $*" | tee -a runs/chain.log; }
note "chain-samples queued, waiting for chain-dose + chain-transfer"
while pgrep -f "[c]hain-(dose|transfer).sh" >/dev/null; do sleep 60; done
ARMS="general,leftover,prompt + template direction (leftover alpha),prompt + leftover minus template (leftover alpha),prompt + shuffled-label (leftover alpha),prompt + random (leftover alpha)"
note "samples start"
timeout 6h $PY -m production.miragelab.dose --run runs/slake --stage test --samples 3 --sample_arms "$ARMS" 2>&1 \
  | grep --line-buffered -vE "Loading weights|Fetching|NvMap" >> runs/dose-samples.log \
  && note "samples ok: $(grep -c sampled runs/dose-samples.log) sampled arms" || note "samples FAILED (runs/dose-samples.log)"
