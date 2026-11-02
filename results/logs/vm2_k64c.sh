#!/bin/bash
# VM2, 64 step weights (3/11 00:10). The loss counts the 5 official steps equally, but the final AdamW displacement
# barely depends on the last ones: lr_k/lr_0 = 1, .905, .655, .345, .095, and through the bias-corrected first moment
# the sign of g_j moves the final weights in the ratio 1, .462, .205, .071, .014 (step 0 sets m-hat alone, then decays).
# K64a = those Adam-influence weights, K64w = --wlr (lr ratios). Same start (D64_1 it180), target and recipe as W64_1
# (val 84.59; K64t tau .15 84.64). Waits for vm2_kt8.sh (3 signmatch jobs at ~20 GB is the safe maximum on VM2).
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "runs/vm2_kt8.sh" > /dev/null; do sleep 60; done
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
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
k K64a --step-w 1,0.462,0.205,0.071,0.014
k K64w --wlr
echo "k64c done $(date +%T)" >> $L
