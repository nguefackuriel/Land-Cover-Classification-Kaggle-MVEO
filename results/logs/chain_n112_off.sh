#!/bin/bash
# after the n=112 T0 run (PID $1): export its best, official baseline.py on 6 seeds
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
while kill -0 $1 2>/dev/null; do sleep 20; done
$PY export_synth.py cache/pfd3_n112_T0_best.pt sets/pfd3_n112_T0 > runs/export_n112_T0.out 2>&1
$PY official_seeds.py sets/pfd3_n112_T0 pfd3_n112_T0 42 0 1 2 3 4 > runs/official_n112_T0.out 2>&1
