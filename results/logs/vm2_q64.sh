#!/bin/bash
# VM2, 64 track (2/11): the 224 recipe that gave KD2 (dither 0.98 + gradient-weighted sign matching) was never run at
# 64 (dither alone was flat: D64_1..3). Serial rounds W64_1..4 from the best 64 start: D64_1 it180 (val 84.30, hold
# 85.21). Log runs/dither_W64_.log.
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f vm2_check.sh > /dev/null; do sleep 30; done
S=cache/sm_D64_1_it180.pt  # projections of KD2 to 64 score 30-43%, so start from the best 64 set
echo "start set $S" >> runs/dither_W64_.log
V=cache/views_n10_s42.pt runs/queue_dither.sh $S W64_ 1 4 0 --weight grad
