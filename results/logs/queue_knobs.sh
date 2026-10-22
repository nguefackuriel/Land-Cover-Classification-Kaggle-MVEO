#!/bin/bash
# Hybrid-round knobs from C2J0H7 at the usual lr 0.2 -> 0.02 (round 8 = same near target, plain hybrid: pooled ~86.28).
# K1 weight delta, K2 weight grad, K3 consensus sign of two near targets (data seeds 0 and 1), K4 near target 2000 steps.
# Then two new lineages (C8J2, C8J3) with hybrid rounds 2..5. usage: queue_knobs.sh [pid to wait for]
# Logs runs/knobs.log, scores runs/pooled_K*.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt
START=cache/sm_C2J0H7_it160.pt
NEAR0=cache/ceiling3_near_C2J0H8_it1000.pt
[ -n "$1" ] && while kill -0 $1 2> /dev/null; do sleep 30; done

pooled() {  # tag start -> exact_eval of the start set and every 20-step checkpoint
  $PY exact_eval.py $V $2 $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$1.log
}
best_of() {
  cat "$@" | python3 -c "
import sys,json; r=[json.loads(l) for l in sys.stdin if l.startswith('{')]; b=max(r,key=lambda d:d['pooled']); print(b['pooled'],b['set'])"
}
hyb() {  # tag target [extra signmatch args]
  T=$1; TG=$2; shift 2
  echo "$T start target $TG $* $(date +%T)" >> runs/knobs.log
  $PY signmatch.py --init $START --views $V --target $TG --iters 160 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --pfd-w 3.0 --real-bs 1024 --tag $T "$@" > runs/signmatch_$T.out 2>&1
  pooled $T $START
  echo "$T done best $(best_of runs/pooled_$T.log) $(date +%T)" >> runs/knobs.log
}
near() {  # tag iters data_seed -> cache/ceiling3_near_<tag>_it<iters>.pt from the end state of START
  [ -f cache/state_K.pt ] || $PY replay_state.py $START $V cache/state_K.pt > runs/loop_K.log 2>&1
  $PY ceiling3.py --T 0 --opt adam --iters $2 --lr 0.02 --bs 256 --eval-every 250 --save-every $2 --data-seed $3 \
      --init-state cache/state_K.pt --tag near_$1 >> runs/loop_K.log 2>&1
}

hyb K1 $NEAR0 --weight delta
hyb K2 $NEAR0 --weight grad
near K3 1000 1
hyb K3 $NEAR0,cache/ceiling3_near_K3_it1000.pt
near K4 2000 0
hyb K4 cache/ceiling3_near_K4_it2000.pt
rm -f cache/state_K.pt

for j in 2 3; do
  runs/lineage2.sh cache/init_c8s1j${j}_sz224.pt $V C8J$j 400
  echo "C8J$j lineage done $(date +%T)" >> runs/knobs.log
  runs/queue_C2J0H.sh cache/sm_C8J${j}H1_best.pt 2 5 C8J${j}H
  echo "C8J$j rounds 2-5 done best $(best_of runs/pooled_C8J${j}H5.log) $(date +%T)" >> runs/knobs.log
done
echo "all done $(date +%T)" >> runs/knobs.log
