#!/bin/bash
# Size curve, second try (11/11): 112 rounds 2-3 from its only good round (sm_SC112r3_it280 = effectively round 1,
# tag SC112b), alone on the VM1 GPU. 224 rounds 2-3 on VM2 (SC224b), 160 rounds 1-3 on VM3 (SC160r).
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "vm1 curve start $(date +%T)" >> runs/vm1_curve.log
V=cache/views_n10_s42_sz112.pt runs/queue_dither_n.sh cache/sm_SC112r3_it280.pt SC112b 2 3 0 $KN
echo "vm1 curve done $(date +%T)" >> runs/vm1_curve.log
