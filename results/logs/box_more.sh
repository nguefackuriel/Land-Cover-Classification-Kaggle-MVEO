#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
export PROTO_PIL=1 PROTO_EXACT=1 DD_HOLDOUT=4096
while pgrep -f scale_more.sh > /dev/null; do sleep 20; done
/home/ubuntu/vllm-env/bin/python box_scan.py cache/views_n10_s42.pt cache/sm_A64R3_it160.pt 1.0,1.05,1.1,1.15,1.2 > runs/box_scan_A64R3.log 2>&1
