#!/bin/bash
# A64: does the full 224 knob set help at 64? Same start (K64a it180) and target (near_N641, 3000 it) as N641
# (val max 84.42), plus --tau 0.15 --pfd-w 5.0 --real-bs 2048. Logs runs/vm2_k64.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=cache/views_n10_s42.pt; L=runs/vm2_k64.log; S=cache/sm_K64a_it180.pt; T=A64
echo "$T start $S near_N641 all knobs + step-w $(date +%T)" >> $L
$PY signmatch.py --init $S --views $V --target cache/ceiling3_near_N641_it3000.pt --iters 200 --opt adam --lr 0.2 \
    --lr-end 0.02 --warmup 10 --pfd-w 3.0 --real-bs 1024 --dither 0.98 --weight grad --tau 0.15 --pfd-w 5.0 \
    --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014 --tag $T > runs/signmatch_$T.out 2>&1
$PY exact_eval.py $V $S $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$T.log
echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$T.log')]; v=max(r,key=lambda d:d['val']); m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
