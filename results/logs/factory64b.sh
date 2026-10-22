#!/bin/bash
# 64x64 track, part 2. P8J0 plateaued after round 4 (pooled 84.3, hold 84.9), so instead of rounds 6..9 on one lineage
# we run more clean lineages with rounds 2..5 each and keep the best: P8J1 (lineage already running), then P8J2..P8J5.
# usage: factory64b.sh <pid of the running P8J1 lineage2.sh>   Progress in runs/factory64.log.
cd /home/ubuntu/Uriel/data_centric/work
export DD_HOLDOUT=4096 V=cache/views_n10_s42.pt
while kill -0 $1 2> /dev/null; do sleep 30; done
echo "P8J1 lineage done $(date +%T)" >> runs/factory64.log
for j in 1 2 3 4 5; do
  [ $j -gt 1 ] && runs/lineage2.sh cache/init_c8s1j${j}.pt $V P8J$j 400 && echo "P8J$j lineage done $(date +%T)" >> runs/factory64.log
  runs/queue_C2J0H.sh cache/sm_P8J${j}H1_best.pt 2 5 P8J${j}H
  echo "P8J$j rounds 2-5 done, best $(cat runs/pooled_P8J${j}H[2-5].log | python3 -c "
import sys, json; r = [json.loads(l) for l in sys.stdin if l.startswith('{')]; b = max(r, key=lambda d: d['pooled']); print(b['set'], b['val'], b['hold'], b['pooled'])") $(date +%T)" >> runs/factory64.log
done
echo "factory64b done $(date +%T)" >> runs/factory64.log
