#!/bin/bash
# VM2 (3/11 03:40). Step weights (--step-w, Adam influence of g_k's sign on theta5) gave the best 64 val so far (K64a
# 84.81 at it180, last-4 mean 84.56 vs 84.33 for W64_1, same start and target). Here:
# 1. official baseline.py check of KT2 it160 (val 87.70 in exact_eval, best 224);
# 2. KTa (224) = KT2 again (start KT1 it140, target near_KT2, all knobs) plus --step-w: clean test at 224;
# 3. W64a (64) = fresh round from K64a it180 with --step-w (the 64 rounds were flat so far).
# Logs runs/vm2_q224.log (KTa) and runs/vm2_k64.log (W64a).
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
SW="--step-w 1,0.462,0.205,0.071,0.014"
$PY export_synth.py cache/sm_KT2_it160.pt sets/n10_KT2i160_224 > runs/vm2_official_KT2i160.log 2>&1
$PY /home/ubuntu/Uriel/data_centric/ext/challenge_2026_public/baseline.py train --data sets/n10_KT2i160_224 --output runs/vm2_official_KT2i160 >> runs/vm2_official_KT2i160.log 2>&1
rm -f runs/vm2_official_KT2i160/best.pt
run() {   # log views tag start target extra...
  L=$1; V=$2; T=$3; S=$4; TG=$5; shift 5
  echo "$T start $S $TG $* $(date +%T)" >> $L
  $PY signmatch.py --init $S --views $V --target $TG --iters 200 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
      --pfd-w 3.0 --real-bs 1024 --dither 0.98 --weight grad --tag $T "$@" > runs/signmatch_$T.out 2>&1
  $PY exact_eval.py $V $S $(ls cache/sm_${T}_it*.pt | sort -V | tail -n +2) 2>/dev/null | grep "^{" > runs/pooled_$T.log
  echo "$T done $(python3 -c "
import json; r=[json.loads(l) for l in open(\"runs/pooled_$T.log\")]; b=max(r,key=lambda d:d[\"pooled\"]); v=max(r,key=lambda d:d[\"val\"])
m=sum(d[\"val\"] for d in r[-4:])/4
print(\"best pooled\", b[\"set\"], b[\"val\"], b[\"hold\"], b[\"pooled\"], \"| best val\", v[\"set\"], v[\"val\"], v[\"hold\"], \"| last4 val %.4f\" % m)") $(date +%T)" >> $L
}
run runs/vm2_q224.log cache/views_n10_s42_sz224.pt KTa cache/sm_KT1_it140.pt cache/ceiling3_near_KT2_it1000.pt \
    --tau 0.15 --pfd-w 5.0 --real-bs 2048 $SW
V=cache/views_n10_s42.pt; S=cache/sm_K64a_it180.pt
$PY replay_state.py $S $V cache/state_W64a.pt > runs/loop_W64a.log 2>&1
$PY ceiling3.py --T 0 --opt adam --iters 1000 --lr 0.02 --bs 256 --eval-every 250 --save-every 1000 --data-seed 0 \
    --init-state cache/state_W64a.pt --tag near_W64a >> runs/loop_W64a.log 2>&1
rm -f cache/state_W64a.pt
run runs/vm2_k64.log $V W64a $S cache/ceiling3_near_W64a_it1000.pt $SW
echo "sw done $(date +%T)" >> runs/vm2_k64.log
