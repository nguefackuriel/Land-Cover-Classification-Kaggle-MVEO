#!/bin/bash
# 12/11: size curve beyond 5 rounds. 160 (SC160c, from its round 5 run on VM2) rounds 6-10, then 112 (SC112b) rounds
# 6-10. Do 160 and 224 end at the same plateau? 224 goes on alone on VM2 (runs/vm2_curve4.sh). One job at a time.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "vm1 curve3 start $(date +%T)" >> runs/vm1_curve.log
V=cache/views_n10_s42_sz160.pt runs/queue_dither_n.sh cache/sm_SC160c5_it280.pt SC160d 6 10 0 $KN
S=$(grep "round 5 done" runs/dither_SC112b.log | tail -1 | awk '{print $5}')
V=cache/views_n10_s42_sz112.pt runs/queue_dither_n.sh $S SC112b 6 10 0 $KN
echo "vm1 curve3 done $(date +%T)" >> runs/vm1_curve.log
