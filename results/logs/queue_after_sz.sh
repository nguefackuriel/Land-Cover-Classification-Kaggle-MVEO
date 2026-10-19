#!/bin/bash
# one GPU job at a time while the QA2026 vLLM job runs: wait for the 224 loop, then the 64px sigmoid-loss test
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "loop_rounds2.sh cache/pfd3_sz_n10_bes[t]" > /dev/null; do sleep 30; done
PROTO_PIL=1 /home/ubuntu/vllm-env/bin/python signmatch.py --init cache/pfd3_pf_r3n_best.pt --views cache/views_n10_s42.pt \
  --target cache/ceiling3_near_L4_it500.pt --iters 160 --opt adam --lr 0.5 --lr-end 0.05 --warmup 20 --loss sigmoid \
  --tau 0.3 --tag sig1 > runs/signmatch_sig1.out 2>&1
