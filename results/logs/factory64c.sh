#!/bin/bash
# After factory64b: rounds 6..9 on the 64x64 lineage with the best val + hold over rounds 2..5 of P8J1..P8J5.
# usage: factory64c.sh <pid of factory64b.sh>   Progress in runs/factory64.log.
cd /home/ubuntu/Uriel/data_centric/work
export DD_HOLDOUT=4096 V=cache/views_n10_s42.pt
while kill -0 $1 2> /dev/null; do sleep 30; done
BEST=$(cat runs/pooled_P8J[0-5]H[2-5].log | python3 -c "
import sys, json; r = [json.loads(l) for l in sys.stdin if l.startswith('{')]; print(max(r, key=lambda d: d['pooled'])['set'])")
T=$(basename $BEST | sed -E 's/sm_(.*H)[0-9]+_it.*/\1/')
echo "64 best after round 5: $BEST ($T), rounds 6-9 $(date +%T)" >> runs/factory64.log
runs/queue_C2J0H.sh $BEST 6 9 $T
echo "$T rounds 6-9 done, best $(cat runs/pooled_${T}[6-9].log | python3 -c "
import sys, json; r = [json.loads(l) for l in sys.stdin if l.startswith('{')]; b = max(r, key=lambda d: d['pooled']); print(b['set'], b['val'], b['hold'], b['pooled'])") $(date +%T)" >> runs/factory64.log
