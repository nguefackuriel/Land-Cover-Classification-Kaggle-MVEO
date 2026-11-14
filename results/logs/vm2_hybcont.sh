#!/bin/bash
# 14/11 05:00: after vm2_hybr1.sh (round 1 of HYC / HYD / HYE), continue the best of the three from round 2 with cont224.sh.
cd /home/ubuntu/Uriel/data_centric/work
until grep -q "hyb r1 done" runs/switch224.log; do sleep 30; done
B=$(for h in C D E; do grep "round 1 done" runs/dither_HY$h.log | tail -1 | awk -v h=$h '{print $7, h, $5}'; done | sort -rn | head -1)
read P H S <<< "$B"
echo "best round 1: HY$H $P ($S), continued from round 2 $(date +%T)" >> runs/switch224.log
runs/cont224.sh $S HY$H 2
