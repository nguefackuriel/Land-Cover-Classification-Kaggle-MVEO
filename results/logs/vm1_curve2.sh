#!/bin/bash
# Size curve, rounds 4-5 (12/11): 112 rounds 4-5 (tag SC112b, from its best round 3), then 64 rounds 4-5 from the best
# 64 round 3 (sm_SC64r3_it300, run on VM2; new tag SC64b so the VM2 log is not overwritten). One job at a time.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "vm1 curve2 start $(date +%T)" >> runs/vm1_curve.log
S=$(grep "round 3 done" runs/dither_SC112b.log | tail -1 | awk '{print $5}')
V=cache/views_n10_s42_sz112.pt runs/queue_dither_n.sh $S SC112b 4 5 0 $KN
V=cache/views_n10_s42.pt runs/queue_dither_n.sh cache/sm_SC64r3_it300.pt SC64b 4 5 0 $KN
echo "vm1 curve2 done $(date +%T)" >> runs/vm1_curve.log
