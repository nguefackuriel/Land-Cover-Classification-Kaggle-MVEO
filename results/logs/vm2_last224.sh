#!/bin/bash
# 10/11: do 224 lineages converge to the same ceiling like the 64 ones did (A64R3, L64P4, L64Q all stop at 84.9-85.0)?
# Full recipe (dither, weight grad, tau .15, pfd-w 5, real-bs 2048, step-w, 300 its, 3000-it near targets) on the two
# best other 224 lineages, at their old-recipe plateau: C8J0H6 it140 (val 85.56 / hold 85.55) and C2J1H5 it160
# (85.09 / 85.42). C2J0 went 86.44 -> 88.47 with the same knobs. 10 rounds each in parallel, must end before Sun 15/11.
# Logs runs/dither_L224C8.log, runs/dither_L224C2.log, scores runs/pooled_L224C8<r>.log.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True V=cache/views_n10_s42_sz224.pt IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
runs/queue_dither_n.sh cache/sm_C8J0H6_it140.pt L224C8 1 10 0 $KN &
sleep 120
runs/queue_dither_n.sh cache/sm_C2J1H5_it160.pt L224C2 1 10 0 $KN &
wait
echo "last224 done $(date +%T)" >> runs/vm2_last.log
