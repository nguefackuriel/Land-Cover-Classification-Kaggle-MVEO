#!/bin/bash
# Repeated rounds on the exact seed-42 run (n=10): near target -> sign matching -> exact PFD polish.
# usage: loop_rounds.sh <start_set.pt> <first_round> <last_round>
cd "$(dirname "$0")/.."
PY=${PY:-python}
SET=$1; R0=$2; R1=$3
V=cache/views_n10_s42.pt
for r in $(seq $R0 $R1); do
  T=L$r
  echo "round $r start $SET $(date +%T)" >> runs/loop_rounds.log
  PROTO_PIL=1 PROTO_EXACT=1 $PY src/replay_state.py $SET $V cache/state_$T.pt > runs/loop_$T.log 2>&1
  $PY src/ceiling3.py --T 0 --opt adam --iters 500 --lr 0.02 --bs 256 --eval-every 100 --save-every 500 \
      --init-state cache/state_$T.pt --tag near_$T >> runs/loop_$T.log 2>&1
  PROTO_PIL=1 PROTO_EXACT=1 $PY src/signmatch.py --init $SET --views $V --target cache/ceiling3_near_${T}_it500.pt --iters 160 --opt adam \
      --lr 0.7 --lr-end 0.07 --warmup 20 --tag $T > runs/signmatch_$T.out 2>&1
  PROTO_PIL=1 PROTO_EXACT=1 $PY src/pfd3.py --init cache/sm_${T}_best.pt --offload --heads 1 --lr 0.02 --lr-end 0.003 --iters 120 \
      --eval-every 20 --outer-T 0 --fixed-head-seed 42 --views $V --views-prob 1.0 --select-exact --fp32-inner \
      --no-seed-eval --tag pf_$T > runs/pfd3_pf_$T.out 2>&1
  SET=cache/pfd3_pf_${T}_best.pt
  echo "round $r done: sm $(grep exact_acc runs/signmatch_$T.out | tail -n +2 | sort -t: -k3 | tail -1 | cut -c1-60) | pf best $(grep exact_acc runs/pfd3_pf_$T.out | python3 -c "
import sys,json; print(max(json.loads(l)['exact_acc'] for l in sys.stdin))") $(date +%T)" >> runs/loop_rounds.log
done
