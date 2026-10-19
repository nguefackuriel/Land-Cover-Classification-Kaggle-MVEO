#!/bin/bash
# one round of "near target + sign matching": replay the set's exact run, find the best reachable network near its end
# point (ceiling3 from that state), then sign-match the images to the corner of that network.
# usage: iterate_sm.sh <set.pt> <round_tag> [sm extra args]
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
SET=$1; TAG=$2; shift 2
PROTO_PIL=1 $PY replay_state.py $SET cache/views_n10_s42.pt cache/state_$TAG.pt > runs/iter_$TAG.log 2>&1
$PY ceiling3.py --T 0 --opt adam --iters 300 --lr 0.02 --bs 256 --eval-every 100 --save-every 300 \
    --init-state cache/state_$TAG.pt --tag near_$TAG >> runs/iter_$TAG.log 2>&1
PROTO_PIL=1 $PY signmatch.py --init $SET --views cache/views_n10_s42.pt --target cache/ceiling3_near_$TAG.pt \
    --iters 200 --opt adam --tag $TAG "$@" > runs/signmatch_$TAG.out 2>&1
