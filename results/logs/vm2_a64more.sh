#!/bin/bash
# A64R rounds 4-6 after round 3 (the 64 lineage climbs: last-4 val 84.74 -> 84.80 -> 84.89). Start = best pooled
# checkpoint of A64R3, ties broken toward the later checkpoint (a tie with the start set re-ran the same round twice).
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
while kill -0 62316 2> /dev/null; do sleep 60; done
S=$(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_A64R3.log')]; m=max(d['pooled'] for d in r)
print([d['set'] for d in r if d['pooled']==m][-1])")
echo "A64R rounds 4-6 from $S $(date +%T)" >> runs/vm2_k64.log
V=cache/views_n10_s42.pt runs/queue_dither.sh $S A64R 4 6 0 --weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 \
    --step-w 1,0.462,0.205,0.071,0.014
echo "A64R more done $(date +%T)" >> runs/vm2_k64.log
