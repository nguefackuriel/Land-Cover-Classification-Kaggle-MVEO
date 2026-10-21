#!/bin/bash
# Lineage screen on the 224 track: four serial hybrid rounds (2..5) on each candidate, then rounds 6..9 on the one
# with the best val + held-out score after round 5. Candidates: C2J1 (lineage already done) and two new lineages from
# 8-cluster k-means medoids (seed 1, pool rows only, make_inits.py). 4096 train rows held out throughout.
# Progress in runs/screen224.log, round scores in runs/pooled_<tag>.log.
cd /home/ubuntu/Uriel/data_centric/work
export DD_HOLDOUT=4096
V=cache/views_n10_s42_sz224.pt
echo "start $(date +%T)" >> runs/screen224.log
runs/queue_C2J0H.sh cache/sm_C2J1H1_best.pt 2 5 C2J1H
echo "C2J1 rounds 2-5 done $(date +%T)" >> runs/screen224.log
for j in 0 1; do
  runs/lineage2.sh cache/init_c8s1j${j}_sz224.pt $V C8J$j 400
  echo "C8J$j lineage done $(date +%T)" >> runs/screen224.log
  runs/queue_C2J0H.sh cache/sm_C8J${j}H1_best.pt 2 5 C8J${j}H
  echo "C8J$j rounds 2-5 done $(date +%T)" >> runs/screen224.log
done
BEST=$(cat runs/pooled_C2J1H5.log runs/pooled_C8J0H5.log runs/pooled_C8J1H5.log | python3 -c "
import sys, json; r = [json.loads(l) for l in sys.stdin if l.startswith('{')]; print(max(r, key=lambda d: d['pooled'])['set'])")
T=$(basename $BEST | sed -E 's/sm_(.*H)[0-9]+_it.*/\1/')
echo "screen best $BEST ($T) $(date +%T)" >> runs/screen224.log
runs/queue_C2J0H.sh $BEST 6 9 $T
echo "done $(date +%T)" >> runs/screen224.log
