#!/bin/bash
# One hybrid round: fresh near target from the start set (NIT its, data seed DS), then IT pixel its with all knobs +
# step-w, every checkpoint scored (val + hold). usage: vm2_round.sh <tag> <start.pt> <views.pt> <NIT> <DS> [IT=300] [wait pid]
# Logs runs/vm2_round.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_round.log; T=$1; S=$2; V=$3; N=$4; DS=$5; IT=${6:-300}; WAIT=$7
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
[ -n "$WAIT" ] && while kill -0 $WAIT 2> /dev/null; do sleep 30; done
echo "$T start $S views $V target $N its seed $DS, $IT its $(date +%T)" >> $L
$PY replay_state.py $S $V cache/state_$T.pt > runs/loop_$T.log 2>&1
$PY ceiling3.py --T 0 --opt adam --iters $N --lr 0.02 --bs 256 --eval-every 1000 --save-every $N --data-seed $DS \
    --init-state cache/state_$T.pt --tag near_$T >> runs/loop_$T.log 2>&1
rm -f cache/state_$T.pt
echo "$T target ready $(grep val_acc runs/loop_$T.log | tail -1) $(date +%T)" >> $L
$PY signmatch.py --init $S --views $V --target cache/ceiling3_near_${T}_it$N.pt --iters $IT --opt adam --lr 0.2 \
    --lr-end 0.02 --warmup 10 --dither 0.98 $KN --tag $T > runs/signmatch_$T.out 2>&1
$PY exact_eval.py $V $S $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$T.log
echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$T.log')]; v=max(r,key=lambda d:d['val']); p=max(r,key=lambda d:d['pooled'])
m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| best pooled', p['set'], p['pooled'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
