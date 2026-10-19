#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
$PY official_seeds.py sets/pfd3_n120_T0s2 pfd3_T0s2 5 6 7 8 > runs/official_T0s2_more.out 2>&1
$PY official_seeds.py sets/pfd3_n120_T01 pfd3_T01 5 6 7 8 > runs/official_T01_more.out 2>&1
