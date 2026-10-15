#!/bin/bash
cd work
PY=python
$PY pfd3.py --init cache/pfd3_n120_T0s2_init.pt --window 20 --offload --heads 2 --lr 0.02 --lr-end 0.005 --iters 150 --outer-T 0 --fp32-inner --tf32 --tag n120_T0_tf32 > runs/pfd3_n120_T0_tf32.out 2>&1
PROTO_PIL=0 $PY fidelity_pil_eval.py cache/pfd3_n120_T0_tf32_best.pt > runs/tf32_6seed.out 2>&1
