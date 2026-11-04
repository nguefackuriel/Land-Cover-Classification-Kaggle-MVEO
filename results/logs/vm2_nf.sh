#!/bin/bash
# NF: stronger exact meta-gradient. pfd-w 3 -> 5 (with real-bs 1024 -> 2048) was one of the knobs that moved 224.
# Same start (NL1 it300) and target (near_NLs1) as NLs1 (val max 88.47, last-4 88.23), 300 its:
#   NF1 = --pfd-w 10 --real-bs 2048 ; NF2 = --pfd-w 10 --real-bs 4096. Logs runs/vm2_q224.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=cache/views_n10_s42_sz224.pt; L=runs/vm2_q224.log; S=cache/sm_NL1_it300.pt
run() {   # tag pfd-w real-bs
  echo "$1 start $S near_NLs1 --pfd-w $2 --real-bs $3 $(date +%T)" >> $L
  $PY signmatch.py --init $S --views $V --target cache/ceiling3_near_NLs1_it3000.pt --iters 300 --opt adam --lr 0.2 \
      --lr-end 0.02 --warmup 10 --dither 0.98 --weight grad --tau 0.15 --step-w 1,0.462,0.205,0.071,0.014 \
      --pfd-w $2 --real-bs $3 --tag $1 > runs/signmatch_$1.out 2>&1
  $PY exact_eval.py $V $S $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$1.log
  echo "$1 done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$1.log')]; v=max(r,key=lambda d:d['val']); p=max(r,key=lambda d:d['pooled'])
m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| best pooled', p['set'], p['pooled'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
}
run NF1 10.0 2048 &
sleep 60; run NF2 10.0 4096 &
wait
echo "nf done $(date +%T)" >> $L
