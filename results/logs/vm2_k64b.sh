#!/bin/bash
# VM2, 64 knobs part 2 (2/11 night), after vm2_k64.sh. K64lr = pixel Adam lr 0.5 -> 0.05 levels instead of 0.2 -> 0.02:
# a 64 pixel feeds ~12 view pixels, and a round only moves pixels by 0.6 level on average, so maybe 64 needs bigger
# steps. Same start (D64_1 it180) and near target as W64_1 (reference val 84.59). Log runs/vm2_k64.log.
cd /home/ubuntu/Uriel/data_centric/work
# (2/11 23:00: started at once in the slot freed by stopping the W64 rounds)
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42.pt; S=cache/sm_D64_1_it180.pt; TG=cache/ceiling3_near_W64_1_it1000.pt; L=runs/vm2_k64.log
k() {
  T=$1; shift
  echo "$T start $* $(date +%T)" >> $L
  $PY signmatch.py --init $S --views $V --target $TG --iters 200 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --pfd-w 3.0 --real-bs 1024 --dither 0.98 --weight grad --tag $T "$@" > runs/signmatch_$T.out 2>&1
  $PY exact_eval.py $V $S $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep "^{" > runs/pooled_$T.log
  echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open(\"runs/pooled_$T.log\")]; b=max(r,key=lambda d:d[\"pooled\"]); v=max(r,key=lambda d:d[\"val\"])
print(\"best pooled\", b[\"set\"], b[\"val\"], b[\"hold\"], b[\"pooled\"], \"| best val\", v[\"set\"], v[\"val\"], v[\"hold\"])") $(date +%T)" >> $L
}
k K64lr --lr 0.5 --lr-end 0.05
echo "k64b done $(date +%T)" >> $L
