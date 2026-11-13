#!/bin/bash
# 13/11: when lineage <old tag> logs "round <r> done", stop it and start a new 224 lineage from <start set> with cont224.sh.
# usage: switch224.sh <old tag> <round> <start set> <new tag>
cd /home/ubuntu/Uriel/data_centric/work
until grep -q "round $2 done" runs/dither_$1.log; do sleep 15; done
pkill -f "cont224.sh [^ ]* $1 "; pkill -f "queue_dither_n.sh [^ ]* $1 "; sleep 2
for s in TERM KILL; do for p in $(pgrep -f "$1"); do [ "$p" != $$ ] && kill -$s $p 2> /dev/null; done; sleep 5; done   # not this script
echo "$1 stopped after round $2, start $4 from $3 $(date +%T)" >> runs/switch224.log
runs/cont224.sh $3 $4 1
