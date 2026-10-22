#!/bin/bash
# After the 224 factory: two more hybrid rounds on lineage C2J0 (never saw the held-out rows), each round's
# checkpoints (it 60..160) scored on val + held-out rows, next round from the best pooled one. Clean test of whether
# extra hybrid rounds help. Results in runs/pooled_C2J0H.log.
# usage: queue_C2J0H.sh [start set] [first round] [last round] [tag]   (default: C2J0H1, rounds 2..3, tag C2J0H)
# Another lineage: queue_C2J0H.sh cache/sm_C2J3H1_best.pt 2 9 C2J3H  (logs runs/pooled_C2J3H*.log)
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096
V=${V:-cache/views_n10_s42_sz224.pt}   # V=cache/views_n10_s42.pt for 64x64 sets
while pgrep -f "runs/factory22[4]" > /dev/null; do sleep 30; done
SET=${1:-cache/sm_C2J0H1_best.pt}
T=${4:-C2J0H}
for r in $(seq ${2:-2} ${3:-3}); do
  runs/loop_hybrid2.sh $SET $V $T $r $r
  PROTO_PIL=1 PROTO_EXACT=1 $PY exact_eval.py $V $SET cache/sm_$T${r}_it{60,80,100,120,140,160}.pt 2>/dev/null | grep '^{' \
      | tee -a runs/pooled_$T.log > runs/pooled_$T$r.log
  SET=$(tail -n +2 runs/pooled_$T$r.log | python3 -c "
import sys, json; r = [json.loads(l) for l in sys.stdin]; print(max(r, key=lambda d: d['pooled'])['set'])")
  echo "round $r best $SET $(date +%T)" >> runs/pooled_$T.log
done
echo "done $(date +%T)" >> runs/pooled_$T.log
