#!/bin/bash
# After box_scan: is "5% further is better" systematic? scale_scan on the 64 set and on older 224 sets.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export PROTO_PIL=1 PROTO_EXACT=1 DD_HOLDOUT=4096
while pgrep -f box_scan.py > /dev/null; do sleep 20; done
$PY scale_scan.py cache/views_n10_s42.pt cache/sm_A64R3_it160.pt > runs/scale_scan_A64R3.log 2>&1
$PY scale_scan.py cache/views_n10_s42_sz224.pt cache/sm_KT2_it160.pt > runs/scale_scan_KT2.log 2>&1
$PY scale_scan.py cache/views_n10_s42_sz224.pt cache/sm_C8J0H6_it140.pt > runs/scale_scan_C8J0H6.log 2>&1
