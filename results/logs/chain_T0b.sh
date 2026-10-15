#!/bin/bash
# waits for the stage-2 python (PID given), then official 6 seeds on its best, then the clean from-medoids T-fitted run
cd work
PY=python
while kill -0 $1 2>/dev/null; do sleep 20; done
$PY export_synth.py cache/pfd3_n120_T0_stage2_best.pt sets/pfd3_n120_T0s2 > runs/export_T0s2.out 2>&1
$PY official_seeds.py sets/pfd3_n120_T0s2 pfd3_T0s2 42 0 1 2 3 4 > runs/official_T0s2.out 2>&1
$PY pfd3.py --init cache/medoids_ipc12.pt --window 20 --offload --heads 2 --iters 300 --outer-T 0 --tag n120_T0_med > runs/pfd3_n120_T0_med.out 2>&1
