#!/bin/bash
# Size curve, 112 / 160 / 224 part, in parallel on VM3 (64 stays on VM2). Same ten medoids (init_c4j0, bicubic) and same
# final recipe as vm2_curve.sh, three rounds each. Logs runs/dither_SC<size>r.log, scores runs/pooled_SC<size>r<round>.log.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "vm3 curve start $(date +%T)" >> runs/vm3_curve.log
for s in 112 160 224; do
  V=cache/views_n10_s42_sz$s.pt runs/queue_dither_n.sh cache/init_c4j0_sz$s.pt SC${s}r 1 3 0 $KN &
  sleep 120
done
wait
echo "vm3 curve done $(date +%T)" >> runs/vm3_curve.log
