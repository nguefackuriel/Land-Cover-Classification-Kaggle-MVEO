#!/bin/bash
# After the SZB lineage: a second hybrid round from the submitted 224 set (SZH1), then pooled scores of the candidates.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
V=cache/views_n10_s42_sz224.pt
while pgrep -f "lineage2.sh cache/init_hyb3_sz22[4]" > /dev/null; do sleep 30; done
runs/loop_hybrid2.sh cache/sm_SZH1_best.pt $V SZH 2 2
PROTO_PIL=1 PROTO_EXACT=1 $PY exact_eval.py $V cache/sm_SZH1_best.pt cache/sm_SZH2_best.pt 2>/dev/null | grep '^{' > runs/pooled_SZH2.log
echo "done $(date +%T)" >> runs/pooled_SZH2.log
