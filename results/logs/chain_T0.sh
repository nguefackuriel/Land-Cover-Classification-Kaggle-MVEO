#!/bin/bash
# after pfd3_n120_T0_fromopt: (1) continue from its best for 200 more its; (2) clean run from the medoids with T fitted
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
# (wait loop removed: pgrep -f matched the launching shell)
cp cache/pfd3_n120_T0_fromopt_best.pt cache/pfd3_n120_T0_stage1.pt
$PY pfd3.py --init cache/pfd3_n120_T0_stage1.pt --window 20 --offload --heads 2 --lr 0.03 --lr-end 0.005 --iters 200 --outer-T 0 --tag n120_T0_stage2 > runs/pfd3_n120_T0_stage2.out 2>&1
$PY pfd3.py --init cache/medoids_ipc12.pt --window 20 --offload --heads 2 --iters 300 --outer-T 0 --tag n120_T0_med > runs/pfd3_n120_T0_med.out 2>&1
