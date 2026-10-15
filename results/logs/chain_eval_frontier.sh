#!/bin/bash
# one GPU job at a time: official multi-seed eval of the submitted set + medoids, then PFD3 frontier points n=96, n=64
cd work
PY=python
$PY official_seeds.py sets/pfd3_n120 pfd3_n120 0 1 2 3 4 2>&1 | grep -v -i warn
$PY official_seeds.py sets/medoids_n120 medoids_n120 0 1 2 3 4 2>&1 | grep -v -i warn
echo "[chain] official seeds done $(date)"
$PY pfd3.py --init cache/medoids_n96.pt --window 20 --iters 300 --heads 2 --offload --tag n96 > runs/pfd3_n96.out 2>&1
echo "[chain] n96 done $(date)"
$PY pfd3.py --init cache/medoids_n64.pt --window 20 --iters 300 --heads 2 --offload --tag n64 > runs/pfd3_n64.out 2>&1
echo "[chain] n64 done $(date)"
