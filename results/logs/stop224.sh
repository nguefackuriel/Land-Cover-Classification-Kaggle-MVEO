#!/bin/bash
# 14/11: stop lineage <tag> once it logs "round <r> done" (same kill as switch224.sh, nothing started after).
# usage: stop224.sh <tag> <round>
cd /home/ubuntu/Uriel/data_centric/work
until grep -q "round $2 done" runs/dither_$1.log; do sleep 15; done
pkill -f "cont224.sh [^ ]* $1 "; pkill -f "queue_dither_n.sh [^ ]* $1 "; sleep 2
for s in TERM KILL; do for p in $(pgrep -f "_$1"); do [ "$p" != $$ ] && kill -$s $p 2> /dev/null; done; sleep 5; done
echo "$1 stopped after round $2 $(date +%T)" >> runs/switch224.log
