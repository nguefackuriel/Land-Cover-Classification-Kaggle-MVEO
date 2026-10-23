#!/bin/bash
# Night of 23/10. From DR3 it160 (submitted, 0.81394), dithered 200-step hybrid rounds with knobs never tried on the
# dithered setup: KD2 weight grad, KD3 consensus sign of the three near targets already computed from DR3 it160 (data
# seeds 0, 1, 2), KD5 heavier PFD term (weight 5, real batch 2048). Then two new 224 lineages (C8J2, C8J3) with four
# dithered rounds each. Judge on val first: on DR3 the held-out block promised +0.7 and the test gave +0.07.
# Logs runs/night.log, scores runs/pooled_<tag>.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt
START=cache/sm_DR3_it160.pt
N0=cache/ceiling3_near_DR4_it1000.pt; N1=cache/ceiling3_near_DS1_it1000.pt; N2=cache/ceiling3_near_DL1_it1000.pt
kd() {  # tag target [extra args]
  T=$1; TG=$2; shift 2
  echo "$T start $* $(date +%T)" >> runs/night.log
  $PY signmatch.py --init $START --views $V --target $TG --iters 200 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --pfd-w 3.0 --real-bs 1024 --dither 0.98 --tag $T "$@" > runs/signmatch_$T.out 2>&1
  $PY exact_eval.py $V $START $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep '^{' > runs/pooled_$T.log
  echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open('runs/pooled_$T.log')]; b=max(r,key=lambda d:d['pooled']); v=max(r,key=lambda d:d['val'])
print('best pooled', b['set'], b['val'], b['hold'], b['pooled'], '| best val', v['set'], v['val'], v['hold'])") $(date +%T)" >> runs/night.log
}
kd KD2 $N0 --weight grad
kd KD3 $N0,$N1,$N2
kd KD5 $N0 --pfd-w 5.0 --real-bs 2048
for j in 2 3; do
  runs/lineage2.sh cache/init_c8s1j${j}_sz224.pt $V C8J$j 400
  echo "C8J$j lineage done $(grep '^{' runs/lineage_C8J$j.log | tail -1) $(date +%T)" >> runs/night.log
  runs/queue_dither.sh cache/sm_C8J${j}H1_best.pt C8J${j}D 2 5 0
  echo "C8J$j dithered rounds done $(tail -2 runs/dither_C8J${j}D.log | head -1) $(date +%T)" >> runs/night.log
done
echo "all done $(date +%T)" >> runs/night.log
