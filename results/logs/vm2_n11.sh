#!/bin/bash
# n=11: one more image for the weakest class (score cost 0.0048 = 0.97 points of test accuracy to win back).
#   E11c = 224, NL1 it300 + a copy of its class-C2 image ; E11m = 224, NL1 it300 + a real class-C2 medoid (bilinear 224)
#   E64c = 64, A64R3 it200 + a copy of its class-C6 image.
# Each: fresh near target (3000 its) from its own n=11 start, then 300 its with all knobs + step-w. Logs runs/vm2_n11.log.
# usage: vm2_n11.sh <C2> <C6>
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_n11.log; C2=$1; C6=$2
V2=cache/views_n11_s42_sz224.pt; V6=cache/views_n11_s42.pt
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
$PY make_n11.py cache/sm_NL1_it300.pt $C2 copy cache/n11_E11c.pt >> $L 2>&1
$PY make_n11.py cache/sm_NL1_it300.pt $C2 medoid cache/n11_E11m.pt >> $L 2>&1
$PY make_n11.py cache/sm_A64R3_it200.pt $C6 copy cache/n11_E64c.pt >> $L 2>&1
$PY exact_eval.py $V2 cache/n11_E11c.pt cache/n11_E11m.pt 2>/dev/null | grep '^{' >> $L
$PY exact_eval.py $V6 cache/n11_E64c.pt 2>/dev/null | grep '^{' >> $L
target() {   # tag start views
  $PY replay_state.py $2 $3 cache/state_$1.pt > runs/loop_$1.log 2>&1
  $PY ceiling3.py --T 0 --opt adam --iters 3000 --lr 0.02 --bs 256 --eval-every 500 --save-every 3000 --data-seed 0 \
      --init-state cache/state_$1.pt --tag near_$1 >> runs/loop_$1.log 2>&1
  rm -f cache/state_$1.pt
  echo "$1 target ready $(date +%T)" >> $L
}
sm() {   # tag start views
  $PY signmatch.py --init $2 --views $3 --target cache/ceiling3_near_$1_it3000.pt --iters 300 --opt adam --lr 0.2 \
      --lr-end 0.02 --warmup 10 --dither 0.98 $KN --tag $1 > runs/signmatch_$1.out 2>&1
  $PY exact_eval.py $3 $2 $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$1.log
  echo "$1 done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$1.log')]; v=max(r,key=lambda d:d['val']); p=max(r,key=lambda d:d['pooled'])
m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| best pooled', p['set'], p['pooled'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
}
echo "n11 start C2=$C2 C6=$C6 $(date +%T)" >> $L
target E11c cache/n11_E11c.pt $V2; sm E11c cache/n11_E11c.pt $V2 &
target E11m cache/n11_E11m.pt $V2; sm E11m cache/n11_E11m.pt $V2 &
target E64c cache/n11_E64c.pt $V6; sm E64c cache/n11_E64c.pt $V6 &
wait
echo "n11 done $(date +%T)" >> $L
