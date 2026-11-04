#!/bin/bash
# Rounds toward "climbed" targets (climb_target.py, 4/11): our own run-end signs with ~1-2% of weights flipped toward the
# near target in small re-linearised chunks (corner 224: 88.84 -> ~90+, 64: 85.75 -> ~87). The flipped weights get
# --flip-w x the weight. CL224 / CL64 = standard knobs; CL64p = same as CL64 with --pfd-w 2 so matching drives more.
# Compare: NLs1 (same start, 88.47 / last-4 88.23), A64T6 (85.01 / 84.93), SC64 (self target, 84.79 / 84.68).
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_cl.log; V2=cache/views_n10_s42_sz224.pt; V6=cache/views_n10_s42.pt
KN="--weight grad --tau 0.15 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
for p in $WAIT; do while kill -0 $p 2> /dev/null; do sleep 30; done; done
pick() {   # saved climb target with the best hold accuracy
  python3 -c "
import json; r=[json.loads(l) for l in open('$1') if '\"it\"' in l]; r=[d for d in r if d['it'] in (5,10,15,20)]
print(max(r, key=lambda d: d['hold'])['it'])"
}
T224=cache/climbNL1_it$(pick runs/climb224.log).pt; T64=cache/climbA64_it$(pick runs/climb64.log).pt
echo "cl start targets $T224 $T64 $(date +%T)" >> $L
run() {   # tag start views target ref pfd-w
  echo "$1 start $2 target $4 pfd-w $6 $(date +%T)" >> $L
  $PY signmatch.py --init $2 --views $3 --target $4 --iters 300 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --dither 0.98 $KN --pfd-w $6 --flip-ref $5 --flip-w 20 --tag $1 > runs/signmatch_$1.out 2>&1
  $PY exact_eval.py $3 $2 $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$1.log
  echo "$1 done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$1.log')]; v=max(r,key=lambda d:d['val']); p=max(r,key=lambda d:d['pooled'])
m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| best pooled', p['set'], p['pooled'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
}
run CL224 cache/sm_NL1_it300.pt $V2 $T224 cache/selfNL1.pt 5.0 &
sleep 60; run CL64 cache/sm_A64R3_it200.pt $V6 $T64 cache/selfA64R3.pt 5.0 &
sleep 60; run CL64p cache/sm_A64R3_it200.pt $V6 $T64 cache/selfA64R3.pt 2.0 &
wait
echo "cl done $(date +%T)" >> $L
