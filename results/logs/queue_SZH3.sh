#!/bin/bash
# Third hybrid round from the submitted 224 set (SZH2), then pooled scores of SZH2 and SZH3.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
V=cache/views_n10_s42_sz224.pt
runs/loop_hybrid2.sh cache/sm_SZH2_best.pt $V SZH 3 3
PROTO_PIL=1 PROTO_EXACT=1 $PY exact_eval.py $V cache/sm_SZH2_best.pt cache/sm_SZH3_best.pt 2>/dev/null | grep '^{' > runs/pooled_SZH3.log
echo "done $(date +%T)" >> runs/pooled_SZH3.log
