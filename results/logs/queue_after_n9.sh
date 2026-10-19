#!/bin/bash
# one job at a time: PFD polish of the submitted hybrid set, then hybrid variants (PFD weight 2 and 5) from the set before it
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
V=cache/views_n10_s42.pt
PROTO_PIL=1 $PY pfd3.py --init cache/sm_hyb3_it120.pt --offload --heads 1 --lr 0.02 --lr-end 0.003 --iters 120 \
    --eval-every 20 --outer-T 0 --fixed-head-seed 42 --views $V --views-prob 1.0 --select-exact --fp32-inner \
    --no-seed-eval --save-every-eval --tag pf_hyb3 > runs/pfd3_pf_hyb3.out 2>&1
for w in 2 5; do
  PROTO_PIL=1 $PY signmatch.py --init cache/pfd3_pf_r3n_best.pt --views $V --target cache/ceiling3_fn_r3n_it1000.pt \
      --iters 150 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 --pfd-w $w --real-bs 1024 --tag hybw$w > runs/signmatch_hybw$w.out 2>&1
done
