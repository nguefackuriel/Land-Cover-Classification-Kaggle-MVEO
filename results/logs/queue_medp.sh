#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "tag [d]w1" > /dev/null; do sleep 30; done
runs/loop_rounds2.sh cache/pfd3_n10_exact_med_best.pt cache/views_n10_s42.pt MEDP 1 4
