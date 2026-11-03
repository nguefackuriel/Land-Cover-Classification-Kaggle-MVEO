#!/bin/bash
# official baseline.py (unmodified, seed 42) on the KT1 candidates: val must match exact_eval
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
for it in 140 160; do
  $PY export_synth.py cache/sm_KT1_it$it.pt sets/n10_KT1i${it}_224 > runs/vm2_official_KT1i$it.log 2>&1
  $PY /home/ubuntu/Uriel/data_centric/ext/challenge_2026_public/baseline.py train --data sets/n10_KT1i${it}_224 --output runs/vm2_official_KT1i$it >> runs/vm2_official_KT1i$it.log 2>&1
  rm -f runs/vm2_official_KT1i$it/best.pt
done
