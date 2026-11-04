#!/bin/bash
# "Small step" targets (flip_target.py, act mode, never run as a full round): keep the sign pattern of our own exact run
# end and take the near target's sign only on the top-f of the disagreeing weights, ranked by first-order gain at our
# corner (2/11 diag on KD2: corner 87.46 -> 88.00 with f = 2%). A trust region instead of a far jump.
#   FA2 / FA1 = 224 from NL1 it300 (near_NLs1, f = 2% / 1%) ; FA64 = 64 from A64R3 it200 (near_A64T6, f = 2%). 300 its.
# Compare: NLs1 (same start + near target, 88.47 / last-4 88.23), A64T6 (85.01 / 84.93). Logs runs/vm2_fa.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_fa.log; V2=cache/views_n10_s42_sz224.pt; V6=cache/views_n10_s42.pt
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "fa start $(date +%T)" >> $L
$PY replay_state.py cache/sm_NL1_it300.pt $V2 cache/state_FA224.pt > /dev/null 2>&1
$PY flip_target.py cache/ceiling3_near_NLs1_it3000.pt cache/state_FA224.pt cache/flipNL1 0.005,0.01,0.02,0.03,0.05 act 2>/dev/null | grep '^{' >> $L
$PY replay_state.py cache/sm_A64R3_it200.pt $V6 cache/state_FA64.pt > /dev/null 2>&1
$PY flip_target.py cache/ceiling3_near_A64T6_it6000.pt cache/state_FA64.pt cache/flipA64 0.005,0.01,0.02,0.03,0.05 act 2>/dev/null | grep '^{' >> $L
rm -f cache/state_FA224.pt cache/state_FA64.pt
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
for p in $WAIT; do while kill -0 $p 2> /dev/null; do sleep 30; done; done   # env WAIT: pids of jobs to let finish
run FA2 cache/sm_NL1_it300.pt $V2 cache/flipNL1_f0.02.pt &
sleep 60; run FA1 cache/sm_NL1_it300.pt $V2 cache/flipNL1_f0.01.pt &
sleep 60; run FA64 cache/sm_A64R3_it200.pt $V6 cache/flipA64_f0.02.pt &
wait
echo "fa done $(date +%T)" >> $L
