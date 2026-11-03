#!/bin/bash
# N64: 64 rounds with a 3000-it near target (at 224 NLK raised the peak from 87.70 to 87.83) + step weights (K64a, best
# 64 val 84.81; W64a, a 1000-it round from it, stayed at 84.72). Rounds N641-2 from K64a it180.
# Logs runs/vm2_k64.log and runs/dither_N64.log.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_k64.log
echo "N64 start cache/sm_K64a_it180.pt NIT=3000 --step-w $(date +%T)" >> $L
V=cache/views_n10_s42.pt NIT=3000 runs/queue_dither_n.sh cache/sm_K64a_it180.pt N64 1 2 0 --weight grad \
    --step-w 1,0.462,0.205,0.071,0.014
echo "N64 done $(date +%T)" >> $L
