#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "tag [p]f_r3n" > /dev/null; do sleep 20; done
runs/loop_rounds.sh cache/pfd3_pf_r3n_best.pt 4 8
