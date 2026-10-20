#!/bin/bash
# Hybrid rounds from the final set (SZH2) with 4096 train rows held out of the near target and the real gradients
# (DD_HOLDOUT). Each round's checkpoints (it 60..160) are scored on val + held-out rows; the next round starts from
# the best pooled one. usage: runs/queue_SZK.sh <rounds>
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096
V=cache/views_n10_s42_sz224.pt
SET=cache/sm_SZH2_best.pt
for r in $(seq 1 ${1:-2}); do
  runs/loop_hybrid2.sh $SET $V SZK $r $r
  PROTO_PIL=1 PROTO_EXACT=1 $PY exact_eval.py $V $SET cache/sm_SZK${r}_it{60,80,100,120,140,160}.pt 2>/dev/null | grep '^{' \
      | tee -a runs/pooled_SZK.log > runs/pooled_SZK$r.log
  SET=$(tail -n +2 runs/pooled_SZK$r.log | python3 -c "
import sys, json; r = [json.loads(l) for l in sys.stdin]; print(max(r, key=lambda d: d['pooled'])['set'])")
  echo "round $r best $SET $(date +%T)" >> runs/pooled_SZK.log
done
echo "done $(date +%T)" >> runs/pooled_SZK.log
