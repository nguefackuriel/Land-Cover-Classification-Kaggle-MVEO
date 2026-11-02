#!/bin/bash
# After vm2_q224.sh: sign matching from KD2 it200 toward a "small step" target = KD2 run corner + the 0.28% flips of
# the near target that help most at our own corner (flip_target.py ... act, f=0.02: corner 87.46 -> 88.00; bigger f hurt).
# Reference: DG1 (same start, full near target DG1) val max 87.11. Log runs/vm2_q224.log.
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "runs/vm2_q224.sh" > /dev/null; do sleep 60; done
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42_sz224.pt; S=cache/sm_KD2_it200.pt; L=runs/vm2_q224.log
for f in 0.02; do
  T=FL$f
  echo "$T start $S target flipA224_f$f $(date +%T)" >> $L
  $PY signmatch.py --init $S --views $V --target cache/flipA224_f$f.pt --iters 200 --opt adam --lr 0.2 --lr-end 0.02 \
      --warmup 10 --pfd-w 3.0 --real-bs 1024 --dither 0.98 --weight grad --tag $T > runs/signmatch_$T.out 2>&1
  $PY exact_eval.py $V $S $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep "^{" > runs/pooled_$T.log
  echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open(\"runs/pooled_$T.log\")]; b=max(r,key=lambda d:d[\"pooled\"]); v=max(r,key=lambda d:d[\"val\"])
print(\"best pooled\", b[\"set\"], b[\"val\"], b[\"hold\"], b[\"pooled\"], \"| best val\", v[\"set\"], v[\"val\"], v[\"hold\"])") $(date +%T)" >> $L
done
