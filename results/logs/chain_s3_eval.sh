#!/bin/bash
cd work
while kill -0 $1 2>/dev/null; do sleep 20; done
PROTO_PIL=0 python fidelity_pil_eval.py cache/pfd3_n120_T0_stage3h4_best.pt > runs/s3_6seed.out 2>&1
