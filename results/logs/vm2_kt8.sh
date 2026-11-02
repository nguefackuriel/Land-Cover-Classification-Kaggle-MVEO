#!/bin/bash
# VM2, 224 (2/11 23:05): tau 0.15 beat 0.3 (KGt val 87.28 vs DG1 87.11, same start KD2 it200 and target near_DG1), so
# try 0.08 the same way (KGt8). Starts when vm2_k64.sh (K64t) frees its GPU slot. Log runs/vm2_q224.log.
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "runs/vm2_k64.sh" > /dev/null; do sleep 60; done
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt; L=runs/vm2_q224.log; S=cache/sm_KD2_it200.pt; T=KGt8
echo "$T start $S near_DG1 --weight grad --tau 0.08 $(date +%T)" >> $L
$PY signmatch.py --init $S --views $V --target cache/ceiling3_near_DG1_it1000.pt --iters 200 --opt adam --lr 0.2 \
    --lr-end 0.02 --warmup 10 --pfd-w 3.0 --real-bs 1024 --dither 0.98 --weight grad --tau 0.08 --tag $T > runs/signmatch_$T.out 2>&1
$PY exact_eval.py $V $S $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep "^{" > runs/pooled_$T.log
echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open(\"runs/pooled_$T.log\")]; b=max(r,key=lambda d:d[\"pooled\"]); v=max(r,key=lambda d:d[\"val\"])
print(\"best pooled\", b[\"set\"], b[\"val\"], b[\"hold\"], b[\"pooled\"], \"| best val\", v[\"set\"], v[\"val\"], v[\"hold\"])") $(date +%T)" >> $L
