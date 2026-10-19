#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
while kill -0 $1 2>/dev/null; do sleep 20; done
PROTO_PIL=0 /home/ubuntu/vllm-env/bin/python fidelity_pil_eval.py cache/pfd3_n120_T0_stage3h4_best.pt > runs/s3_6seed.out 2>&1
