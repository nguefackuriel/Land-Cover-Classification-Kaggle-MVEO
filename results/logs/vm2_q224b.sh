#!/bin/bash
# VM2, 224 track, replaces the tail of vm2_q224.sh (2/11 evening). KG5 (PFD weight 5, real batch 2048) gave val 87.14,
# hold 87.62 from KD2 it200 (KD2: 87.01 / 87.13), so the next rounds start from KG5 it200 with those knobs:
# KR1 = usual round (fresh 1000-it near target), NLK = same start and knobs but a 3000-it near target (is a longer
# target worth it? compare with KR1), then FL0.02 (small-step flip target from KD2, see flip_target.py).
# Logs runs/vm2_q224.log (one line per run) and runs/dither_KR.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt; L=runs/vm2_q224.log
KNOBS="--weight grad --pfd-w 5.0 --real-bs 2048"
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
# KGt was started by vm2_q224.sh (now stopped): wait for its signmatch, then score it
while kill -0 12341 2> /dev/null; do sleep 30; done
summ KGt cache/sm_KD2_it200.pt
S=cache/sm_KG5_it200.pt
runs/queue_dither.sh $S KR 1 1 0 $KNOBS
echo "KR1 $(tail -2 runs/dither_KR.log | head -1)" >> $L
$PY replay_state.py $S $V cache/state_NLK.pt > runs/loop_NLK.log 2>&1
$PY ceiling3.py --T 0 --opt adam --iters 3000 --lr 0.02 --bs 256 --eval-every 500 --save-every 3000 --data-seed 0 \
    --init-state cache/state_NLK.pt --tag near_NLK >> runs/loop_NLK.log 2>&1
rm -f cache/state_NLK.pt
sm NLK $S cache/ceiling3_near_NLK_it3000.pt $KNOBS
sm FL0.02 cache/sm_KD2_it200.pt cache/flipA224_f0.02.pt --weight grad
echo "all done $(date +%T)" >> $L
