#!/bin/bash
# 12/11: the fresh 224 lineage from the medoids (SC224b) is still climbing at round 5 (87.87 pooled, NL1 is 88.53),
# so it goes on, rounds 6-12, same recipe, one job on the GPU. Also extends the size curve.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "curve4 start $(date +%T)" >> runs/vm2_curve.log
S=$(grep "round 5 done" runs/dither_SC224b.log | tail -1 | awk '{print $5}')
V=cache/views_n10_s42_sz224.pt runs/queue_dither_n.sh $S SC224b 6 12 0 $KN
echo "curve4 done $(date +%T)" >> runs/vm2_curve.log
