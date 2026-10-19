#!/bin/bash
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
$PY diag_step.py --init cache/pfd3_n120_best.pt --dir-heads 4 --kinds lp4,lp8,lp16,lp64 --tag _lp_opt > runs/diag_step_lp_opt.out 2>&1
$PY diag_step.py --init cache/medoids_ipc12.pt --dir-heads 4 --kinds lp4,lp8,lp16,lp64 --tag _lp_med > runs/diag_step_lp_med.out 2>&1
