#!/bin/bash
# after the TF32 chain (PID $1): fixed low temperature T=0.1 from the stage-3 best, then 6-seed replica
cd work
PY=python
while kill -0 $1 2>/dev/null; do sleep 20; done
cp cache/pfd3_n120_T0_stage3h4_best.pt cache/pfd3_n120_s3_init.pt
$PY pfd3.py --init cache/pfd3_n120_s3_init.pt --window 20 --offload --heads 2 --lr 0.02 --lr-end 0.005 --iters 100 --outer-T 0.1 --tag n120_T01 > runs/pfd3_n120_T01.out 2>&1
PROTO_PIL=0 $PY fidelity_pil_eval.py cache/pfd3_n120_T01_best.pt > runs/T01_6seed.out 2>&1
