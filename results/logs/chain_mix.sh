#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
while kill -0 $1 2>/dev/null; do sleep 20; done
/home/ubuntu/vllm-env/bin/python basin_mix.py cache/pfd3_n120_T01_best.pt cache/pfd3_km1_T01_best.pt km0km1 > runs/mix_km0km1.out 2>&1
