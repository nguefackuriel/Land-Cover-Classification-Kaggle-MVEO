#!/bin/bash
# 9/11: sign loss focused on the weights the images can toggle (fragility.py maps, 4/11: a +-1 level change on half
# the pixels flips the net sign of 8.5% (224) / 13% (64) of the weights at least once in 8 copies). Weights outside
# the map get x0.1. Same start and target as the reference rounds, so the comparison is direct:
#   MK224: NL1 it300 + near_NLs1 (3000 it), 300 its   vs NLs1 (88.47 max, last-4 88.23)
#   MK64 : A64R3 it200 + near_A64T6 (6000 it), 300 its vs A64T6 (85.01 max, last-4 84.93)
# Logs runs/vm2_mask.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_mask.log
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
run() {   # tag start views target mask
  echo "$1 start $2 target $4 mask $5 $(date +%T)" >> $L
  $PY signmatch.py --init $2 --views $3 --target $4 --iters 300 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --dither 0.98 $KN --mask $5 --mask-out 0.1 --tag $1 > runs/signmatch_$1.out 2>&1
  $PY exact_eval.py $3 $2 $(ls cache/sm_$1_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$1.log
  echo "$1 done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$1.log')]; v=max(r,key=lambda d:d['val']); p=max(r,key=lambda d:d['pooled'])
m=sum(d['val'] for d in r[-4:])/4
print('best val', v['set'], v['val'], v['hold'], '| best pooled', p['set'], p['pooled'], '| last4 val %.4f' % m)") $(date +%T)" >> $L
}
run MK224 cache/sm_NL1_it300.pt cache/views_n10_s42_sz224.pt cache/ceiling3_near_NLs1_it3000.pt cache/fragNL1.pt
run MK64 cache/sm_A64R3_it200.pt cache/views_n10_s42.pt cache/ceiling3_near_A64T6_it6000.pt cache/fragA64R3.pt
echo "mask done $(date +%T)" >> $L
