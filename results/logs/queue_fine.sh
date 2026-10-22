#!/bin/bash
# Finer hybrid rounds from C2J0H7 (the submitted set). Rounds at lr 0.2 first drop the exact run by ~3 points and
# then climb back, so stage 1 tries lower lr and a heavier PFD term on the same near target (the one round 8 used,
# computed from sm_C2J0H7_it160). Stage 2 chains fresh rounds with the variant that has the best pooled score.
# usage: queue_fine.sh   (logs runs/pooled_F.log, runs/fine.log)
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt
START=cache/sm_C2J0H7_it160.pt
NEAR0=cache/ceiling3_near_C2J0H8_it1000.pt

# name lr lr_end pfd_w real_bs iters
VARIANTS="F1 0.05 0.005 3 1024 160
F2 0.02 0.002 3 1024 160
F3 0.05 0.005 10 2048 160"

pooled() {  # tag, start set -> exact_eval of the start set and every 20-step checkpoint, appended to pooled_F.log
  $PY exact_eval.py $V $2 $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' \
      | tee -a runs/pooled_F.log > runs/pooled_$1.log
}
best_of() {  # pooled_*.log files -> "pooled set" of the best line
  cat "$@" | python3 -c "
import sys,json; r=[json.loads(l) for l in sys.stdin if l.startswith('{')]; b=max(r,key=lambda d:d['pooled']); print(b['pooled'],b['set'])"
}
round() {  # tag start target lr lr_end pfd_w real_bs iters
  echo "$1 start $2 lr $4-$5 pfd_w $6 real_bs $7 iters $8 $(date +%T)" >> runs/fine.log
  $PY signmatch.py --init $2 --views $V --target $3 --iters $8 --opt adam --lr $4 --lr-end $5 --warmup 10 \
      --pfd-w $6 --real-bs $7 --tag $1 > runs/signmatch_$1.out 2>&1
  pooled $1 $2
  echo "$1 done best $(best_of runs/pooled_$1.log) $(date +%T)" >> runs/fine.log
}

echo "$VARIANTS" | while read N LR LE W RB IT; do
  round $N $START $NEAR0 $LR $LE $W $RB $IT < /dev/null
done

# stage 2: serial rounds with the best variant, a fresh near target each round
read P0 SET <<< "$(best_of runs/pooled_F1.log runs/pooled_F2.log runs/pooled_F3.log)"
N=$(echo $SET | sed -nE 's/.*sm_(F[0-9])_.*/\1/p'); N=${N:-F1}
read _ LR LE W RB IT <<< "$(echo "$VARIANTS" | grep "^$N ")"
echo "stage 2 with $N from $SET (pooled $P0) $(date +%T)" >> runs/fine.log
for r in 1 2 3 4; do
  T=FS$r
  $PY replay_state.py $SET $V cache/state_$T.pt > runs/loop_$T.log 2>&1
  $PY ceiling3.py --T 0 --opt adam --iters 1000 --lr 0.02 --bs 256 --eval-every 250 --save-every 1000 \
      --init-state cache/state_$T.pt --tag near_$T >> runs/loop_$T.log 2>&1
  rm -f cache/state_$T.pt
  round $T $SET cache/ceiling3_near_${T}_it1000.pt $LR $LE $W $RB $IT
  read _ SET <<< "$(best_of runs/pooled_$T.log)"
done
echo "all done $(date +%T)" >> runs/fine.log
