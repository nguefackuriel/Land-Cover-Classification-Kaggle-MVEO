#!/bin/bash
# Size curve, rounds 3-5 (12/11). 160 round 3 crashed on VM3 (a vLLM job took 66 GB of its GPU at 00:17), so 160 goes
# on here from its best round 2 (sm_SC160r2_it280, tag SC160c, rounds 3-5), then 224 rounds 4-5 (tag SC224b).
# One job at a time. 112 and 64 rounds 4-5 run on VM1 (runs/vm1_curve2.sh).
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "curve3 start $(date +%T)" >> runs/vm2_curve.log
V=cache/views_n10_s42_sz160.pt runs/queue_dither_n.sh cache/sm_SC160r2_it280.pt SC160c 3 5 0 $KN
S=$(grep "round 3 done" runs/dither_SC224b.log | tail -1 | awk '{print $5}')
V=cache/views_n10_s42_sz224.pt runs/queue_dither_n.sh $S SC224b 4 5 0 $KN
echo "curve3 done $(date +%T)" >> runs/vm2_curve.log
