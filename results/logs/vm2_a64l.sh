#!/bin/bash
# A64R rounds 4-6 with 300 pixel iterations (NL1 at 224: 300 its gave val 88.47 vs ~88.15, still rising at the end).
# Round 4 reuses the near target already trained by the stopped 200-it queue (ceiling3 PID 72171).
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=cache/views_n10_s42.pt; L=runs/dither_A64R.log; S=cache/sm_A64R3_it200.pt; T=A64R4
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
while kill -0 72171 2> /dev/null; do sleep 30; done
rm -f cache/state_$T.pt
echo "round 4 restarted with --iters 300 on near_A64R4 $(date +%T)" >> $L
$PY signmatch.py --init $S --views $V --target cache/ceiling3_near_${T}_it1000.pt --iters 300 --opt adam --lr 0.2 \
    --lr-end 0.02 --warmup 10 --pfd-w 3.0 --real-bs 1024 --dither 0.98 --tag $T $KN > runs/signmatch_$T.out 2>&1
$PY exact_eval.py $V $S $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$T.log
S=$(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$T.log')]; m=max(d['pooled'] for d in r)
print([d['set'] for d in r if d['pooled']==m][-1])")
echo "round 4 done best $S $(date +%T)" >> $L
IT=300 V=$V runs/queue_dither.sh $S A64R 5 6 0 $KN
echo "A64R more done $(date +%T)" >> runs/vm2_k64.log
