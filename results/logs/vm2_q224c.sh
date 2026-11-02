#!/bin/bash
# VM2, 224 track (2/11 22:20), replaces vm2_q224b.sh. KGt (tau 0.15, from KD2 it200, DG1 target) gave val 87.28 (it180
# and it200), hold 87.70: best 224 val so far (KG5 87.14, KD2 87.01). KGc = all knobs together (tau 0.15 + PFD weight 5 +
# real batch 2048), same start and target, to see if they stack. Then rounds KT1, KT2 (fresh near target each) from the
# better of KGt / KGc with its knobs, then NLK (3000-it near target) from the best set. FL0.02 dropped (small-step
# flip target, corner only 88.0). Logs runs/vm2_q224.log and runs/dither_KT.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt; L=runs/vm2_q224.log
summ() {   # tag start
  $PY exact_eval.py $V $2 $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep "^{" > runs/pooled_$1.log
  echo "$1 done $(python3 -c "
import json; r=[json.loads(l) for l in open(\"runs/pooled_$1.log\")]; b=max(r,key=lambda d:d[\"pooled\"]); v=max(r,key=lambda d:d[\"val\"])
print(\"best pooled\", b[\"set\"], b[\"val\"], b[\"hold\"], b[\"pooled\"], \"| best val\", v[\"set\"], v[\"val\"], v[\"hold\"])") $(date +%T)" >> $L
}
sm() {     # tag start target extra...
  T=$1; S=$2; TG=$3; shift 3
  echo "$T start $S $TG $* $(date +%T)" >> $L
  $PY signmatch.py --init $S --views $V --target $TG --iters 200 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --pfd-w 3.0 --real-bs 1024 --dither 0.98 --tag $T "$@" > runs/signmatch_$T.out 2>&1
  summ $T $S
}
ALL="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048"
sm KGc cache/sm_KD2_it200.pt cache/ceiling3_near_DG1_it1000.pt $ALL
# pick start and knobs: the run with the higher mean val over its last 4 checkpoints (less noisy than one max)
read S K <<< "$(python3 -c "
import json
def m(t):
    r=[json.loads(l) for l in open(f'runs/pooled_{t}.log')][-4:]; return sum(d['val'] for d in r)/len(r)
def best(t):
    r=[json.loads(l) for l in open(f'runs/pooled_{t}.log')]; return max(r,key=lambda d:d['pooled'])['set']
print(best('KGc'), 'all') if m('KGc') > m('KGt') else print(best('KGt'), 'tau')")"
[ "$K" = all ] && KN="$ALL" || KN="--weight grad --tau 0.15"
echo "KT start $S knobs $KN $(date +%T)" >> $L
runs/queue_dither.sh $S KT 1 2 0 $KN
echo "KT $(grep done runs/dither_KT.log | tail -2 | tr '\n' ' ')" >> $L
S=$(grep "round 2 done" runs/dither_KT.log | tail -1 | awk '{print $5}')
$PY replay_state.py $S $V cache/state_NLK.pt > runs/loop_NLK.log 2>&1
$PY ceiling3.py --T 0 --opt adam --iters 3000 --lr 0.02 --bs 256 --eval-every 500 --save-every 3000 --data-seed 0 \
    --init-state cache/state_NLK.pt --tag near_NLK >> runs/loop_NLK.log 2>&1
rm -f cache/state_NLK.pt
sm NLK $S cache/ceiling3_near_NLK_it3000.pt $KN
echo "all done $(date +%T)" >> $L
