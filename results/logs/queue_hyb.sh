#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "tag [h]yb4" > /dev/null; do sleep 30; done
runs/loop_hybrid.sh cache/sm_hyb3_it120.pt HY 1 3
