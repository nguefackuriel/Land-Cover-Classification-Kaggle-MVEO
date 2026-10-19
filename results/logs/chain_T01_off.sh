#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
$PY export_synth.py cache/pfd3_n120_T01_best.pt sets/pfd3_n120_T01 > runs/export_T01.out 2>&1
$PY official_seeds.py sets/pfd3_n120_T01 pfd3_T01 42 0 1 2 3 4 > runs/official_T01.out 2>&1
