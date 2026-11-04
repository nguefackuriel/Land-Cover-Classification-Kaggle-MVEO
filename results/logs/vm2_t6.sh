#!/bin/bash
# Three tests, 300 its each with all knobs + step-w, each from its own fresh near target:
#   E11c  = n=11, NL1 it300 + copy of its PermCrop image, rows ordered so epoch-0 views match n=10 (it0 val 44.6), target 3000 its
#   NT6   = 224, NL1 it300, near target of 6000 its (3000 raised the peak over 1000), data seed 2
#   A64T6 = 64, A64R3 it200, near target of 6000 its, data seed 8
# usage: vm2_t6.sh <which>   (E11c | NT6 | A64T6), logs runs/vm2_t6.log
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_t6.log
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
case $1 in
  E11c)  S=cache/n11o_E11c.pt;      V=cache/views_n11_s42_sz224.pt; N=3000; DS=0 ;;
  NT6)   S=cache/sm_NL1_it300.pt;   V=cache/views_n10_s42_sz224.pt; N=6000; DS=2 ;;
  A64T6) S=cache/sm_A64R3_it200.pt; V=cache/views_n10_s42.pt;       N=6000; DS=8 ;;
esac
T=$1
echo "$T start $S target $N its seed $DS $(date +%T)" >> $L
$PY replay_state.py $S $V cache/state_$T.pt > runs/loop_$T.log 2>&1
$PY ceiling3.py --T 0 --opt adam --iters $N --lr 0.02 --bs 256 --eval-every 1000 --save-every $N --data-seed $DS \
    --init-state cache/state_$T.pt --tag near_$T >> runs/loop_$T.log 2>&1
rm -f cache/state_$T.pt
echo "$T target ready $(grep -i val runs/loop_$T.log | tail -1) $(date +%T)" >> $L
$PY signmatch.py --init $S --views $V --target cache/ceiling3_near_${T}_it$N.pt --iters 300 --opt adam --lr 0.2 \
    --lr-end 0.02 --warmup 10 --dither 0.98 $KN --tag $T > runs/signmatch_$T.out 2>&1
$PY exact_eval.py $V $S $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$T.log
echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$T.log')]; v=max(r,key=lambda d:d['val']); p=max(r,key=lambda d:d['pooled'])
m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| best pooled', p['set'], p['pooled'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
