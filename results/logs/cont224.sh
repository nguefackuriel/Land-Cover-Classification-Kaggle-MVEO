#!/bin/bash
# 13/11: carry on the best start sets of the 224 screening (runs/screen224b.sh) with the final recipe, round after
# round, until stopped by hand (GPU work ends Sunday ~15:00 UTC). usage: cont224.sh <start set> <tag> <first round>
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
V=cache/views_n10_s42_sz224.pt runs/queue_dither_n.sh $1 $2 $3 30 0 $KN
