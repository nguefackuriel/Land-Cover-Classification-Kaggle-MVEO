#!/bin/bash
# Size curve, second try (11/11): 160 alone on the GPU, rounds 1-3 from the medoids (all three crashed on 10/11 with
# out-of-memory errors while 112 and 224 shared the GPU). 224 and 112 rounds 2-3 run on VM2 (runs/vm2_curve2.sh).
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "vm3 curve2 start $(date +%T)" >> runs/vm3_curve.log
V=cache/views_n10_s42_sz160.pt runs/queue_dither_n.sh cache/init_c4j0_sz160.pt SC160r 1 3 0 $KN
echo "vm3 curve2 done $(date +%T)" >> runs/vm3_curve.log
