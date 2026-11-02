#!/bin/bash
# VM2, 64 knobs (2/11): with 12x fewer pixels than 224, capacity is scarcer at 64, so knobs that focus it may pay more
# here than at 224. Same start (D64_1 it180) and same near target as round W64_1 (the reference: pow 1, PFD weight 3,
# tau 0.3): K64p2 = --weight-pow 2, K64f6 = PFD weight 6, K64t = tau 0.15. Judge on val. Log runs/vm2_k64.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
V=cache/views_n10_s42.pt; S=cache/sm_D64_1_it180.pt; TG=cache/ceiling3_near_W64_1_it1000.pt; L=runs/vm2_k64.log
while [ ! -f $TG ]; do sleep 60; done; sleep 30
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
k K64p2 --weight-pow 2
k K64f6 --pfd-w 6.0
k K64t --tau 0.15
echo "all done $(date +%T)" >> $L
