#!/bin/bash
# VM2, 224 track (2/11): the two knobs queue_kg.sh never ran (KG5 = PFD weight 5 with real batch 2048, KGt = sign
# temperature 0.15), same start KD2 it200 and same near target as DG1, then a round with a 3x longer near target
# (3000 it) from DG2 it200. Judge on val (test tracks val). Log runs/vm2_q224.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt
L=runs/vm2_q224.log
kg() {   # tag start target extra...
  T=$1; S=$2; TG=$3; shift 3
  echo "$T start $S $* $(date +%T)" >> $L
  $PY signmatch.py --init $S --views $V --target $TG --iters 200 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --pfd-w 3.0 --real-bs 1024 --dither 0.98 --weight grad --tag $T "$@" > runs/signmatch_$T.out 2>&1
  $PY exact_eval.py $V $S $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep "^{" > runs/pooled_$T.log
  echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open(\"runs/pooled_$T.log\")]; b=max(r,key=lambda d:d[\"pooled\"]); v=max(r,key=lambda d:d[\"val\"])
print(\"best pooled\", b[\"set\"], b[\"val\"], b[\"hold\"], b[\"pooled\"], \"| best val\", v[\"set\"], v[\"val\"], v[\"hold\"])") $(date +%T)" >> $L
}
N0=cache/ceiling3_near_DG1_it1000.pt
kg KG5 cache/sm_KD2_it200.pt $N0 --pfd-w 5.0 --real-bs 2048
kg KGt cache/sm_KD2_it200.pt $N0 --tau 0.15
S=cache/sm_DG2_it200.pt
$PY replay_state.py $S $V cache/state_NL.pt > runs/loop_NL.log 2>&1
$PY ceiling3.py --T 0 --opt adam --iters 3000 --lr 0.02 --bs 256 --eval-every 500 --save-every 3000 --data-seed 0 \
    --init-state cache/state_NL.pt --tag near_NL >> runs/loop_NL.log 2>&1
rm -f cache/state_NL.pt
kg NL1 $S cache/ceiling3_near_NL_it3000.pt
echo "all done $(date +%T)" >> $L
