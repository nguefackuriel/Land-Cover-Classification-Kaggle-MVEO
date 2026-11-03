#!/bin/bash
# Two 224 tests from NL1 it300 (val 88.47; NL2 400 it and NL2x 600 it on target near_NL2 only tie it):
#  NP1  = polish: same target near_NL2, low lr 0.05 -> 0.005, 100 iterations (every round gains most in its last,
#         annealed steps; does that tail alone add anything?)
#  NLs1 = fresh round with a new near-target data seed (1), 3000 it, then 300 pixel iterations
# Logs runs/vm2_q224.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=cache/views_n10_s42_sz224.pt; L=runs/vm2_q224.log; S=cache/sm_NL1_it300.pt
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
summ() {
  $PY exact_eval.py $V $S $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$1.log
  echo "$1 done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$1.log')]; v=max(r,key=lambda d:d['val']); p=max(r,key=lambda d:d['pooled'])
m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| best pooled', p['set'], p['pooled'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
}
( echo "NP1 start $S near_NL2 lr .05->.005 100 it $(date +%T)" >> $L
  $PY signmatch.py --init $S --views $V --target cache/ceiling3_near_NL2_it3000.pt --iters 100 --opt adam --lr 0.05 \
      --lr-end 0.005 --warmup 10 --pfd-w 3.0 --real-bs 1024 --dither 0.98 --tag NP1 $KN > runs/signmatch_NP1.out 2>&1
  summ NP1 ) &
( echo "NLs1 start $S near-target seed 1 $(date +%T)" >> $L
  $PY replay_state.py $S $V cache/state_NLs1.pt > runs/loop_NLs1.log 2>&1
  $PY ceiling3.py --T 0 --opt adam --iters 3000 --lr 0.02 --bs 256 --eval-every 500 --save-every 3000 --data-seed 1 \
      --init-state cache/state_NLs1.pt --tag near_NLs1 >> runs/loop_NLs1.log 2>&1
  rm -f cache/state_NLs1.pt
  $PY signmatch.py --init $S --views $V --target cache/ceiling3_near_NLs1_it3000.pt --iters 300 --opt adam --lr 0.2 \
      --lr-end 0.02 --warmup 10 --pfd-w 3.0 --real-bs 1024 --dither 0.98 --tag NLs1 $KN > runs/signmatch_NLs1.out 2>&1
  summ NLs1 ) &
wait
echo "np done $(date +%T)" >> $L
