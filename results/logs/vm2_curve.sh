#!/bin/bash
# Size curve for the 224 justification and the paper: the same ten medoids (init_c4j0) at 64 / 112 / 160 / 224 pixels,
# the same final recipe (dither, weight grad, tau 0.15, pfd-w 5, real batches of 2048, step weights, 3000-it near
# targets, 300 iterations), three rounds each. Rounds are interleaved across sizes so the curve fills in round by round.
# Logs runs/dither_SC<size>r.log, scores runs/pooled_SC<size>r<round>.log. usage: vm2_curve.sh [pid to wait for]
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
WAIT=$1
[ -n "$WAIT" ] && [ "$WAIT" != 0 ] && while kill -0 $WAIT 2> /dev/null; do sleep 60; done
declare -A SET VW
SET[64]=cache/init_c4j0.pt; VW[64]=cache/views_n10_s42.pt
for s in 112 160 224; do SET[$s]=cache/init_c4j0_sz$s.pt; VW[$s]=cache/views_n10_s42_sz$s.pt; done
echo "curve start $(date +%T)" >> runs/vm2_curve.log
for r in 1 2 3; do
  for s in 64 112 160 224; do
    V=${VW[$s]} runs/queue_dither_n.sh ${SET[$s]} SC${s}r $r $r 0 $KN
    SET[$s]=$(grep "round $r done" runs/dither_SC${s}r.log | tail -1 | awk '{print $5}')
    echo "size $s round $r best ${SET[$s]} $(grep "round $r done" runs/dither_SC${s}r.log | tail -1 | awk '{print $7}') $(date +%T)" >> runs/vm2_curve.log
  done
done
echo "curve done $(date +%T)" >> runs/vm2_curve.log
