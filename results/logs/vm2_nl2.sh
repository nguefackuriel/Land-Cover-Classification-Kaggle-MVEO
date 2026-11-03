#!/bin/bash
# NL2: next 224 round from NL1 it300 (val 88.47) with a fresh 3000-it near target, at two lengths sharing that
# target: NL2 = 400 pixel iterations, NL2x = 600. Logs runs/vm2_q224.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
V=cache/views_n10_s42_sz224.pt; L=runs/vm2_q224.log; S=cache/sm_NL1_it300.pt
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "NL2 start $S $(date +%T)" >> $L
$PY replay_state.py $S $V cache/state_NL2.pt > runs/loop_NL2.log 2>&1
$PY ceiling3.py --T 0 --opt adam --iters 3000 --lr 0.02 --bs 256 --eval-every 500 --save-every 3000 --data-seed 0 \
    --init-state cache/state_NL2.pt --tag near_NL2 >> runs/loop_NL2.log 2>&1
rm -f cache/state_NL2.pt
run() {   # tag iters
  $PY signmatch.py --init $S --views $V --target cache/ceiling3_near_NL2_it3000.pt --iters $2 --opt adam --lr 0.2 \
      --lr-end 0.02 --warmup 10 --pfd-w 3.0 --real-bs 1024 --dither 0.98 --tag $1 $KN > runs/signmatch_$1.out 2>&1
  $PY exact_eval.py $V $S $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$1.log
  echo "$1 done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$1.log')]; v=max(r,key=lambda d:d['val']); p=max(r,key=lambda d:d['pooled'])
m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| best pooled', p['set'], p['pooled'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
}
run NL2 400 &
sleep 120; run NL2x 600 &
wait
echo "NL2 all done $(date +%T)" >> $L
