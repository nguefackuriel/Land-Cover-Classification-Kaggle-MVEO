#!/bin/bash
# 13/11: the 224 ceiling follows the starting images, not the history of the lineage: c4j0 88.57 (two histories: NL1,
# SC224b), c8s1j0 87.34 (F224C8; L224C8 87.17), c4j1 86.57 (F224C2; L224C2 86.51). At round 3 the order was already
# right (86.77 / 86.12 / 85.24). So: screen untested medoid sets with 3 rounds of the final recipe, then carry on with
# the best. usage: screen224b.sh <init tag> ... (pairs), one lineage after the other on this GPU.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
while [ $# -ge 2 ]; do
  V=cache/views_n10_s42_sz224.pt runs/queue_dither_n.sh $1 $2 1 3 0 $KN
  echo "$2 ($1) after 3 rounds: $(grep 'round 3 done' runs/dither_$2.log | tail -1 | awk '{print $5, $7}') $(date +%T)" >> runs/screen224b.log
  shift 2
done
echo "screen done $(date +%T)" >> runs/screen224b.log
