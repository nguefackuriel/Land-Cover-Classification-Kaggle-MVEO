#!/bin/bash
# 14/11 17:45: HYC r10 reached 88.76 pooled (val 88.77 / hold 88.75), above NL1 (88.53). Official check of the new best
# sets with the unmodified baseline.py (seed 42), then carry on the lineage. usage: final_off.sh <tag> <next round> <start> <set> ...
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
TAG=$1; R=$2; START=$3; shift 3
for X in "$@"; do
  $PY export_synth.py cache/sm_$X.pt sets/$X > runs/export_$X.log 2>&1
  $PY official_seeds.py sets/$X $X 42 > runs/offcheck_$X.log 2>&1
done
runs/cont224.sh $START $TAG $R
