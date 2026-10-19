#!/bin/bash
# one GPU job at a time: official 6-seed check of 1.5x delta, then PFD3 from two other k-means inits
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
$PY make_x15.py 2>&1 | grep -v -i warn
$PY export_synth.py cache/pfd3_n120_x15.pt sets/pfd3_n120_x15 2>&1 | grep -v -i warn
$PY official_seeds.py sets/pfd3_n120_x15 pfd3_n120_x15 42 0 1 2 3 4 2>&1 | grep -v -i warn
echo "[chain] x15 official done $(date)"
CUDA_VISIBLE_DEVICES= $PY make_medoids_km.py 1 2 2>&1 | grep -v -i warn
for s in 1 2; do
  $PY pfd3.py --init cache/medoids_n120_km$s.pt --window 20 --iters 300 --heads 2 --offload --tag n120_km$s > runs/pfd3_n120_km$s.out 2>&1
  echo "[chain] km$s done $(date)"
done
