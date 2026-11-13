#!/bin/bash
# 13/11: whole-image swaps between 224 lineage ends from different starts (one class image at a time).
cd /home/ubuntu/Uriel/data_centric/work
export PROTO_PIL=1 PROTO_EXACT=1 DD_HOLDOUT=4096
PY=/home/ubuntu/vllm-env/bin/python; V=cache/views_n10_s42_sz224.pt
$PY crossover.py $V cache/sm_NL1_it300.pt cache/sm_S224J68_it280.pt cache/sm_S224K58_it300.pt cache/sm_SC224b11_it260.pt > runs/xover224_NL1.log 2> runs/xover224.err
$PY crossover.py $V cache/sm_S224J68_it280.pt cache/sm_NL1_it300.pt > runs/xover224_J68.log 2>> runs/xover224.err
echo "xover done $(date +%T)" >> runs/xover224_J68.log
