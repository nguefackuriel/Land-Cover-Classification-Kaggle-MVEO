#!/bin/bash
# climb_target.py restricted to weights the run barely commits to (|theta5 - theta0| / R below q): are high-value flips
# also among the easy ones? 64 only (A64R3 it200 run end, near_A64T6), q = 0.5 and 0.25.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
for p in $WAIT; do while kill -0 $p 2> /dev/null; do sleep 30; done; done
$PY climb_target.py cache/selfA64R3.pt cache/climbA64u50 --near cache/ceiling3_near_A64T6_it6000.pt --undecided 0.5 --chunk 0.001 --iters 20 --save 10,20 > runs/climb64u50.log 2> /dev/null
$PY climb_target.py cache/selfA64R3.pt cache/climbA64u25 --near cache/ceiling3_near_A64T6_it6000.pt --undecided 0.25 --chunk 0.001 --iters 20 --save 10,20 > runs/climb64u25.log 2> /dev/null
