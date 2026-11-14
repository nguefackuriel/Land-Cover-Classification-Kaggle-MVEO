#!/bin/bash
# Stop lineage <tag> at a given UTC time (GPU work must end Sunday ~15:00 UTC). usage: stop_at.sh <tag> "2026-11-15 14:30"
cd /home/ubuntu/Uriel/data_centric/work
E=$(date -ud "$2" +%s); while [ $(date +%s) -lt $E ]; do sleep 60; done
pkill -f "final_off.sh $1 "; pkill -f "cont224.sh [^ ]* $1 "; pkill -f "queue_dither_n.sh [^ ]* $1 "; sleep 2
for s in TERM KILL; do for p in $(pgrep -f "_$1"); do [ "$p" != $$ ] && kill -$s $p 2> /dev/null; done; sleep 5; done
echo "$1 stopped at $(date +%T) (time limit)" >> runs/switch224.log
