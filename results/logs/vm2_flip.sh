#!/bin/bash
# Minimal-flip targets (flip_target.py): val accuracy of corners that keep our run end signs except the top-f most
# useful flips of the near target. 224: KD2 it200 end state + near_DG1 (computed from KD2). 64: D64_1 it180 + near_W64_1.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
L=runs/vm2_flip.log
$PY replay_state.py cache/sm_KD2_it200.pt cache/views_n10_s42_sz224.pt cache/state_FT224.pt >> $L 2>&1
$PY flip_target.py cache/ceiling3_near_DG1_it1000.pt cache/state_FT224.pt cache/flip224 0.05,0.1,0.2,0.35,0.5,1 2>&1 | grep "^{" >> $L
while [ ! -f cache/ceiling3_near_W64_1_it1000.pt ]; do sleep 60; done; sleep 30
$PY replay_state.py cache/sm_D64_1_it180.pt cache/views_n10_s42.pt cache/state_FT64.pt >> $L 2>&1
$PY flip_target.py cache/ceiling3_near_W64_1_it1000.pt cache/state_FT64.pt cache/flip64 0.05,0.1,0.2,0.35,0.5,1 2>&1 | grep "^{" >> $L
echo "done $(date +%T)" >> $L
