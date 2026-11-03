#!/bin/bash
# A64R rounds 5-7, 300 pixel iterations, near-target data seed = round number: when a round does not beat its start
# (A64R4: best pooled stayed A64R3 it200), the next round still gets a new target instead of repeating the same one.
# Start of each round = best pooled of the previous one (start set included), ties toward the later checkpoint.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
S=cache/sm_A64R3_it200.pt
for r in 5 6 7; do
  NS=$r IT=300 V=cache/views_n10_s42.pt runs/queue_dither.sh $S A64R $r $r 0 $KN
  S=$(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_A64R$r.log')]; m=max(d['pooled'] for d in r)
print([d['set'] for d in r if d['pooled']==m][-1])")
  echo "round $r start for next: $S (seed $r) $(date +%T)" >> runs/dither_A64R.log
done
echo "A64R seeds done $(date +%T)" >> runs/vm2_k64.log
