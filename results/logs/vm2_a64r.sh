#!/bin/bash
# A64R: 64 rounds with the full 224 knob set + step weights (A64: val 84.89, last-4 84.74 vs 84.40 for N641 on the
# same start and target, still rising at it200). Rounds 1-3 from A64 it200, 1000-it near targets.
# Logs runs/vm2_k64.log and runs/dither_A64R.log.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
echo "A64R start cache/sm_A64_it200.pt $(date +%T)" >> runs/vm2_k64.log
V=cache/views_n10_s42.pt runs/queue_dither.sh cache/sm_A64_it200.pt A64R 1 3 0 --weight grad --tau 0.15 --pfd-w 5.0 \
    --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014
echo "A64R done $(date +%T)" >> runs/vm2_k64.log
