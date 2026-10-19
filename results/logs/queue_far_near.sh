#!/bin/bash
# after sig1: a stronger, farther near target from the 83.88 set's end point (1500 steps, lr .03, PIL eval), then sign matching
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
while pgrep -f "tag [s]ig1" > /dev/null || pgrep -f "tag [p]f_SZ2" > /dev/null; do sleep 20; done
PROTO_PIL=1 $PY ceiling3.py --T 0 --opt adam --iters 1500 --lr 0.03 --bs 256 --eval-every 250 --save-every 500 \
    --init-state cache/state_pf_r3n.pt --tag fn_r3n > runs/ceiling3_fn_r3n.out 2>&1
for it in 1000 1500; do
  PROTO_PIL=1 $PY signmatch.py --init cache/pfd3_pf_r3n_best.pt --views cache/views_n10_s42.pt \
    --target cache/ceiling3_fn_r3n_it$it.pt --iters 160 --opt adam --lr 0.7 --lr-end 0.07 --warmup 20 --tag fn$it > runs/signmatch_fn$it.out 2>&1
done
