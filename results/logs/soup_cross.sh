#!/bin/bash
# 12/11: two independent 224 lineages (NL1 and the fresh SC224b) end at the same ~88.5. Does a weight-space mix of
# their end states do better than each (same-lineage soups gave nothing on 10/11)? Runs after the official check.
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "official_seeds.py sets/SC224b11i260" > /dev/null; do sleep 30; done
PROTO_PIL=1 PROTO_EXACT=1 DD_HOLDOUT=4096 /home/ubuntu/vllm-env/bin/python soup_eval.py cache/views_n10_s42_sz224.pt \
  cache/sm_NL1_it300.pt cache/sm_SC224b11_it260.pt cache/sm_SC224b9_it220.pt cache/sm_SC224b10_it260.pt 2>&1 | grep "^{" > runs/soup_cross.log
