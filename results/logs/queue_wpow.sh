#!/bin/bash
# Weight power for the grad-weighted sign loss, from KD2 it200 with DG1's near target (DG1 = power 1 on the same start
# and target: val 87.11 max, pooled 87.23): WP2 power 2, WP05 power 0.5. Judge on val. Logs runs/wpow.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt
START=cache/sm_KD2_it200.pt; TG=cache/ceiling3_near_DG1_it1000.pt
for spec in "WP2 2" "WP05 0.5"; do
  read T P <<< "$spec"
  echo "$T start pow $P $(date +%T)" >> runs/wpow.log
  $PY signmatch.py --init $START --views $V --target $TG --iters 200 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --pfd-w 3.0 --real-bs 1024 --dither 0.98 --weight grad --weight-pow $P --tag $T > runs/signmatch_$T.out 2>&1
  $PY exact_eval.py $V $START $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$T.log
  echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$T.log')]; b=max(r,key=lambda d:d['pooled']); v=max(r,key=lambda d:d['val'])
print('best pooled', b['set'], b['val'], b['hold'], b['pooled'], '| best val', v['set'], v['val'], v['hold'])") $(date +%T)" >> runs/wpow.log
done
echo "all done $(date +%T)" >> runs/wpow.log
