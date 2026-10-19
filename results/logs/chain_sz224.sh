#!/bin/bash
# exact seed-42 trajectory PFD with 224x224 images (no upsampling in the official RRC), n=1 and n=5
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
while pgrep -f chain_small_n.sh > /dev/null; do sleep 30; done
run() {  # tag init n iters lr lr_end
  PROTO_PIL=1 $PY pfd3.py --init cache/init_$2.pt --offload --heads 1 --lr $5 --lr-end $6 --iters $4 --eval-every 20 \
    --outer-T 0 --fixed-head-seed 42 --views cache/views_n$3_s42_sz224.pt --views-prob 1.0 --select-exact --fp32-inner \
    --no-seed-eval --save-every-eval --tag $1 > runs/pfd3_$1.out 2>&1
}
run sz_n1_river n1_river_sz224 1 300 0.05 0.005
run sz_n5_hard n5_hard_sz224 5 300 0.05 0.005
