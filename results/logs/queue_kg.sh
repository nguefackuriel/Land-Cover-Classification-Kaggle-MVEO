#!/bin/bash
# More knobs on top of dither + grad weight, from KD2 it200 (reference: DG1 = same start, near target NS 0, val max
# 87.11, pooled 87.23). KG3 consensus sign of 3 near targets (data seeds 0, 1, 2), KG5 PFD weight 5 with real batch 2048,
# KGt sign temperature tau 0.15. Judge on val. Logs runs/kg.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt
START=cache/sm_KD2_it200.pt; N0=cache/ceiling3_near_DG1_it1000.pt
kg() {
  T=$1; TG=$2; shift 2
  echo "$T start $* $(date +%T)" >> runs/kg.log
  $PY signmatch.py --init $START --views $V --target $TG --iters 200 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --pfd-w 3.0 --real-bs 1024 --dither 0.98 --weight grad --tag $T "$@" > runs/signmatch_$T.out 2>&1
  $PY exact_eval.py $V $START $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$T.log
  echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$T.log')]; b=max(r,key=lambda d:d['pooled']); v=max(r,key=lambda d:d['val'])
print('best pooled', b['set'], b['val'], b['hold'], b['pooled'], '| best val', v['set'], v['val'], v['hold'])") $(date +%T)" >> runs/kg.log
}
$PY replay_state.py $START $V cache/state_KG.pt > runs/loop_KG.log 2>&1
for s in 1 2; do
  $PY ceiling3.py --T 0 --opt adam --iters 1000 --lr 0.02 --bs 256 --eval-every 250 --save-every 1000 --data-seed $s \
      --init-state cache/state_KG.pt --tag near_KGs$s >> runs/loop_KG.log 2>&1
done
rm -f cache/state_KG.pt
kg KG3 $N0,cache/ceiling3_near_KGs1_it1000.pt,cache/ceiling3_near_KGs2_it1000.pt
kg KG5 $N0 --pfd-w 5.0 --real-bs 2048
kg KGt $N0 --tau 0.15
echo "all done $(date +%T)" >> runs/kg.log
