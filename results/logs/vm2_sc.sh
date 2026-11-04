#!/bin/bash
# Self-corner targets. The pure-sign corner of our own exact run end beats the run itself (224 NL1 it300: 88.84 vs 88.47;
# 64 A64R3 it200: 85.75 vs 85.01): weights whose step gradients change sign stop inside the box (Adam momentum cancels).
# Sign matching to s_act = sign(theta5 - theta0) (the replay state itself is a valid --target) asks all 5 steps to agree
# with the run's own final signs. SC2a = 224 self ; SC2b = 224 self + top 0.5% act-ranked flips of near_NLs1 (corner
# 89.06) ; SC64 = 64 self. 300 its, all knobs. Compare NLs1 (88.47 / last-4 88.23), A64T6 (85.01 / 84.93).
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_sc.log; V2=cache/views_n10_s42_sz224.pt; V6=cache/views_n10_s42.pt
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
for p in $WAIT; do while kill -0 $p 2> /dev/null; do sleep 30; done; done
echo "sc start $(date +%T)" >> $L
$PY replay_state.py cache/sm_NL1_it300.pt $V2 cache/selfNL1.pt > /dev/null 2>&1
$PY replay_state.py cache/sm_A64R3_it200.pt $V6 cache/selfA64R3.pt > /dev/null 2>&1
run() {   # tag start views target
  echo "$1 start $2 target $4 $(date +%T)" >> $L
  $PY signmatch.py --init $2 --views $3 --target $4 --iters 300 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --dither 0.98 $KN --tag $1 > runs/signmatch_$1.out 2>&1
  $PY exact_eval.py $3 $2 $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$1.log
  echo "$1 done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$1.log')]; v=max(r,key=lambda d:d['val']); p=max(r,key=lambda d:d['pooled'])
m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| best pooled', p['set'], p['pooled'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
}
run SC2a cache/sm_NL1_it300.pt $V2 cache/selfNL1.pt &
sleep 60; run SC2b cache/sm_NL1_it300.pt $V2 cache/flipNL1_f0.005.pt &
sleep 60; run SC64 cache/sm_A64R3_it200.pt $V6 cache/selfA64R3.pt &
wait
echo "sc done $(date +%T)" >> $L
