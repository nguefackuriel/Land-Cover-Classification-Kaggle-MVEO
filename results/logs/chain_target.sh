#!/bin/bash
# n=10 64px exact seed-42: outer gradient pulls theta_K toward a better reachable network (near target from our theta_5)
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
while pgrep -f "ceiling3.py --[T] 0 --opt adam --iters 300" > /dev/null; do sleep 20; done
run() {  # tag target mode w iters
  PROTO_PIL=1 $PY pfd3.py --init cache/pfd3_n10_exact_s2_it280.pt --offload --heads 1 --lr 0.02 --lr-end 0.003 --iters $5 \
    --eval-every 20 --outer-T 0 --fixed-head-seed 42 --views cache/views_n10_s42.pt --views-prob 1.0 --select-exact \
    --fp32-inner --no-seed-eval --save-every-eval --target $2 --target-mode $3 --target-w $4 --tag $1 > runs/pfd3_$1.out 2>&1
}
run tg_near_l2 cache/ceiling3_near_s2.pt l2 -1 200
