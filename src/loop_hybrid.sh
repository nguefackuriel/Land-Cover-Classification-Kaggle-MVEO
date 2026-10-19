#!/bin/bash
# Hybrid rounds on the exact seed-42 run (n=10): fresh near target (PIL eval, 1000 steps) -> sign matching + exact PFD
# meta-gradient (weight 3). usage: loop_hybrid.sh <start_set.pt> <prefix> <first_round> <last_round>
cd "$(dirname "$0")/.."
PY=${PY:-python}
SET=$1; PFX=$2; R0=$3; R1=$4
V=cache/views_n10_s42.pt
for r in $(seq $R0 $R1); do
  T=${PFX}$r
  echo "round $r start $SET $(date +%T)" >> runs/loop_${PFX}.log
  PROTO_PIL=1 PROTO_EXACT=1 $PY src/replay_state.py $SET $V cache/state_$T.pt > runs/loop_$T.log 2>&1
  PROTO_PIL=1 PROTO_EXACT=1 $PY src/ceiling3.py --T 0 --opt adam --iters 1000 --lr 0.02 --bs 256 --eval-every 250 --save-every 1000 \
      --init-state cache/state_$T.pt --tag near_$T >> runs/loop_$T.log 2>&1
  PROTO_PIL=1 PROTO_EXACT=1 $PY src/signmatch.py --init $SET --views $V --target cache/ceiling3_near_${T}_it1000.pt --iters 160 --opt adam \
      --lr 0.2 --lr-end 0.02 --warmup 10 --pfd-w 3.0 --real-bs 1024 --tag $T > runs/signmatch_$T.out 2>&1
  SET=cache/sm_${T}_best.pt
  echo "round $r done: $(grep exact_acc runs/signmatch_$T.out | python3 -c "
import sys,json; r=[json.loads(l) for l in sys.stdin]; print(' '.join('%d:%.4f'%(d['it'],d['exact_acc']) for d in r))") $(date +%T)" >> runs/loop_${PFX}.log
done
