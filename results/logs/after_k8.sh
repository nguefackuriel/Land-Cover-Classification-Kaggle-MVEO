#!/bin/bash
# Waits for K8 in queue_knobs2.sh, stops that queue (K2..K4 and the 224 lineages wait), then chains dithered rounds
# from the best of K6/K7/K8, with --disp when the best one used it.
cd /home/ubuntu/Uriel/data_centric/work
until grep -q "^K8 done" runs/knobs.log; do sleep 20; done
for p in $(pgrep -f "runs/queue_knob[s]2.sh"); do pkill -P $p; kill $p; done
sleep 3; pkill -f "tag K2\b"
read P SET <<< "$(cat runs/pooled_K6.log runs/pooled_K7.log runs/pooled_K8.log | python3 -c "
import sys,json; r=[json.loads(l) for l in sys.stdin if l.startswith('{')]; b=max(r,key=lambda d:d['pooled']); print(b['pooled'],b['set'])")"
EXTRA=""; case $SET in *sm_K7_*|*sm_K8_*) EXTRA="--disp";; esac
echo "after K8: best $SET pooled $P, dithered rounds DR1..DR6 $EXTRA $(date +%T)" >> runs/knobs.log
exec runs/queue_dither.sh $SET DR 1 6 0 $EXTRA
