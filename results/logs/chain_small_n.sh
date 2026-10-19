#!/bin/bash
# exact seed-42 trajectory PFD at n < 10 (inits = subsets of the submitted n=10 set)
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
run() {  # tag init n iters lr lr_end
  PROTO_PIL=1 $PY pfd3.py --init cache/init_$2.pt --offload --heads 1 --lr $5 --lr-end $6 --iters $4 --eval-every 20 \
    --outer-T 0 --fixed-head-seed 42 --views cache/views_n$3_s42.pt --views-prob 1.0 --select-exact --fp32-inner \
    --no-seed-eval --save-every-eval --tag $1 > runs/pfd3_$1.out 2>&1
}
run sm_n5_hard n5_hard 5 400 0.05 0.005
run sm_n1_river n1_river 1 300 0.05 0.005
run sm_n3_a n3_a 3 300 0.05 0.005
run sm_n5_easy n5_easy 5 400 0.05 0.005
