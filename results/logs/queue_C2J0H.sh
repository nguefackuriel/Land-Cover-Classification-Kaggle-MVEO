#!/bin/bash
# After the 224 factory: two more hybrid rounds on lineage C2J0 (never saw the held-out rows), each round's
# checkpoints (it 60..160) scored on val + held-out rows, next round from the best pooled one. Clean test of whether
# extra hybrid rounds help. Results in runs/pooled_C2J0H.log.
# usage: queue_C2J0H.sh [start set] [first round] [last round]   (default: C2J0H1, rounds 2..3)
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096
V=cache/views_n10_s42_sz224.pt
while pgrep -f "runs/factory22[4]" > /dev/null; do sleep 30; done
SET=${1:-cache/sm_C2J0H1_best.pt}
for r in $(seq ${2:-2} ${3:-3}); do
  runs/loop_hybrid2.sh $SET $V C2J0H $r $r
  PROTO_PIL=1 PROTO_EXACT=1 $PY exact_eval.py $V $SET cache/sm_C2J0H${r}_it{60,80,100,120,140,160}.pt 2>/dev/null | grep '^{' \
      | tee -a runs/pooled_C2J0H.log > runs/pooled_C2J0H$r.log
  SET=$(tail -n +2 runs/pooled_C2J0H$r.log | python3 -c "
import sys, json; r = [json.loads(l) for l in sys.stdin]; print(max(r, key=lambda d: d['pooled'])['set'])")
  echo "round $r best $SET $(date +%T)" >> runs/pooled_C2J0H.log
done
echo "done $(date +%T)" >> runs/pooled_C2J0H.log
