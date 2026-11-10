#!/bin/bash
# 64 part of the size curve on VM2: wait for round 1 (started by vm2_curve.sh, whose loop was stopped when 112/160/224
# moved to VM3), then rounds 2-3 from its best. usage: vm2_curve64.sh <pid of the round-1 queue_dither_n.sh>
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
while kill -0 $1 2> /dev/null; do sleep 60; done
SET=$(grep "round 1 done" runs/dither_SC64r.log | tail -1 | awk '{print $5}')
V=cache/views_n10_s42.pt runs/queue_dither_n.sh $SET SC64r 2 3 0 $KN
echo "curve64 done $(date +%T)" >> runs/vm2_curve.log
