#!/bin/bash
# Serial hybrid rounds with a dithered start (signmatch.py --dither 0.98): a round no longer loses ground in its first
# 20 iterations (K6: exact run 85.2 at it20 instead of 82.6), so rounds can be longer and chained. Each round: fresh near
# target from the current set, 200 iterations, every 20-step checkpoint and the start set scored on val + held-out rows,
# next round from the best. usage: queue_dither.sh <start set> <tag> <first round> <last round> [pid to wait for] [extra args]
# Logs runs/dither_<tag>.log, scores runs/pooled_<tag><r>.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=${V:-cache/views_n10_s42_sz224.pt}; IT=${IT:-200}; NS=${NS:-0}   # env: views (64: cache/views_n10_s42.pt), iterations, near-target data seed
SET=$1; TAG=$2; R0=$3; R1=$4; WAIT=$5; shift 5
[ -n "$WAIT" ] && [ "$WAIT" != 0 ] && while kill -0 $WAIT 2> /dev/null; do sleep 30; done
for r in $(seq $R0 $R1); do
  T=$TAG$r
  echo "round $r start $SET $* $(date +%T)" >> runs/dither_$TAG.log
  $PY replay_state.py $SET $V cache/state_$T.pt > runs/loop_$T.log 2>&1
  $PY ceiling3.py --T 0 --opt adam --iters 1000 --lr 0.02 --bs 256 --eval-every 250 --save-every 1000 --data-seed $NS \
      --init-state cache/state_$T.pt --tag near_$T >> runs/loop_$T.log 2>&1
  rm -f cache/state_$T.pt
  $PY signmatch.py --init $SET --views $V --target cache/ceiling3_near_${T}_it1000.pt --iters $IT --opt adam --lr 0.2 \
      --lr-end 0.02 --warmup 10 --pfd-w 3.0 --real-bs 1024 --dither 0.98 --tag $T "$@" > runs/signmatch_$T.out 2>&1
  $PY exact_eval.py $V $SET $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$T.log
  read P SET <<< "$(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$T.log')]; b=max(r,key=lambda d:d['pooled']); print(b['pooled'],b['set'])")"
  echo "round $r done best $SET pooled $P $(date +%T)" >> runs/dither_$TAG.log
done
echo "done $(date +%T)" >> runs/dither_$TAG.log
