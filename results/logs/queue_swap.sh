#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "tag [h]yb3" > /dev/null; do sleep 30; done
runs/loop_rounds2.sh cache/init_swapRes_River.pt cache/views_n10_s42.pt SWP 1 4
