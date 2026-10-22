#!/bin/bash
# 64x64 track (the organizers give 64x64 sets priority in the post-challenge checks, 22/10). Serial hybrid rounds with
# 4096 held-out train rows, as on the 224 track:
#   1. rounds 2..5 from the submitted 64x64 set hyb3 (0.80567; its lineage saw the held-out rows, so its hold is biased)
#   2. two clean lineages from pool-only 8-cluster medoids (P8J0, P8J1), lineage2 + rounds 2..5
#   3. rounds 6..9 on the clean lineage with the best val + hold after round 5
# First waits for the F3 pooled scores of queue_fine.sh, then stops that queue (its stage 2 is 224 work).
# Progress in runs/factory64.log, round scores in runs/pooled_<tag>.log.
cd /home/ubuntu/Uriel/data_centric/work
export DD_HOLDOUT=4096 V=cache/views_n10_s42.pt
until grep -q "^F3 done" runs/fine.log; do sleep 20; done
for p in $(pgrep -f "runs/queue_fin[e].sh"); do pkill -P $p; kill $p; done
sleep 5; pkill -f "tag FS[1-4]\b"; pkill -f "near_FS[1-4]"; pkill -f "state_FS[1-4].pt"
echo "start $(date +%T) (queue_fine stopped after F3)" >> runs/factory64.log

runs/queue_C2J0H.sh cache/sm_hyb3_it120.pt 2 5 M64H
echo "M64H rounds 2-5 done $(date +%T)" >> runs/factory64.log
for j in 0 1; do
  runs/lineage2.sh cache/init_c8s1j${j}.pt $V P8J$j 400
  echo "P8J$j lineage done $(date +%T)" >> runs/factory64.log
  runs/queue_C2J0H.sh cache/sm_P8J${j}H1_best.pt 2 5 P8J${j}H
  echo "P8J$j rounds 2-5 done $(date +%T)" >> runs/factory64.log
done
BEST=$(cat runs/pooled_P8J0H5.log runs/pooled_P8J1H5.log | python3 -c "
import sys, json; r = [json.loads(l) for l in sys.stdin if l.startswith('{')]; print(max(r, key=lambda d: d['pooled'])['set'])")
T=$(basename $BEST | sed -E 's/sm_(.*H)[0-9]+_it.*/\1/')
echo "clean best $BEST ($T) $(date +%T)" >> runs/factory64.log
runs/queue_C2J0H.sh $BEST 6 9 $T
echo "done $(date +%T)" >> runs/factory64.log
