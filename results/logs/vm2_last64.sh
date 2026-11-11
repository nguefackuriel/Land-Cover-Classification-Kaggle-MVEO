#!/bin/bash
# Last week (9/11): full recipe (dither, weight grad, tau .15, pfd-w 5, real-bs 2048, step-w, 300 its, 3000-it near
# targets) on the two best other 64 lineages: Q4J0 (medoids c4j0, pooled 84.85 after 5 old-recipe rounds) and P8J3
# (pooled 84.59 after 4). The same recipe took the hyb3 lineage from ~84.6 to 85.04 val (A64R3, our submitted 64).
# 4 rounds each, in parallel. Logs runs/dither_L64Q.log, runs/dither_L64P.log, scores runs/pooled_L64Q<r>.log.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True V=cache/views_n10_s42.pt IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
runs/queue_dither_n.sh cache/sm_Q4J0G5_it200.pt L64Q 1 4 0 $KN &
sleep 120
runs/queue_dither_n.sh cache/sm_P8J3H4_it160.pt L64P 1 4 0 $KN &
wait
echo "last64 done $(date +%T)" >> runs/vm2_last.log
