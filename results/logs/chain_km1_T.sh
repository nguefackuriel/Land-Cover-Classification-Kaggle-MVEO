#!/bin/bash
# second basin: km1 CE optimum -> T fitted (300 its) -> T=0.1 (100 its) -> 6-seed replica of both
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
while kill -0 $1 2>/dev/null; do sleep 20; done
cp cache/pfd3_n120_km1_best.pt cache/pfd3_km1_ce_init.pt
$PY pfd3.py --init cache/pfd3_km1_ce_init.pt --window 20 --offload --heads 2 --lr 0.03 --lr-end 0.005 --iters 300 --outer-T 0 --tag km1_T0 > runs/pfd3_km1_T0.out 2>&1
cp cache/pfd3_km1_T0_best.pt cache/pfd3_km1_T0_init.pt
$PY pfd3.py --init cache/pfd3_km1_T0_init.pt --window 20 --offload --heads 2 --lr 0.02 --lr-end 0.005 --iters 100 --outer-T 0.1 --tag km1_T01 > runs/pfd3_km1_T01.out 2>&1
PROTO_PIL=0 $PY fidelity_pil_eval.py cache/pfd3_km1_T0_best.pt cache/pfd3_km1_T01_best.pt > runs/km1_6seed.out 2>&1
