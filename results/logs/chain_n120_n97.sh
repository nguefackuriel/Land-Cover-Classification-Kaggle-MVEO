#!/bin/bash
# waits for the n120 PFD3 run, then: export + official seed-42 eval, then PFD3 at n=97 (one GPU job at a time)
cd work
PY=python
while kill -0 7248 2>/dev/null; do sleep 20; done
echo "[chain] n120 run finished $(date)"
$PY export_synth.py cache/pfd3_n120_best.pt sets/pfd3_n120 2>&1 | grep -v -i warn
$PY make_submission.py sets/pfd3_n120 pfd3_n120 2>&1 | grep -v -i warn
echo "[chain] official done $(date)"
$PY pfd3.py --init cache/medoids_n97.pt --window 20 --iters 300 --heads 2 --offload --tag n97 > runs/pfd3_n97.out 2>&1
echo "[chain] n97 finished $(date)"
