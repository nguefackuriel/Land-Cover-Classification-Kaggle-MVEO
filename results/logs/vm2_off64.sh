#!/bin/bash
# official baseline.py (unmodified, seed 42) on one set: usage vm2_off.sh <tag> <it>  (cache/sm_<tag>_it<it>.pt)
# exports sets/n10_<tag>i<it>_64, writes runs/vm2_official_<tag>i<it>.log and a line in runs/vm2_q224.log
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python; N=$1i$2
$PY export_synth.py cache/sm_$1_it$2.pt sets/n10_${N}_64 > runs/vm2_official_$N.log 2>&1
$PY /home/ubuntu/Uriel/data_centric/ext/challenge_2026_public/baseline.py train --data sets/n10_${N}_64 \
    --output runs/vm2_official_$N >> runs/vm2_official_$N.log 2>&1
rm -f runs/vm2_official_$N/best.pt
echo "official $1 it$2 (64): $(grep validation_accuracy runs/vm2_official_$N.log | tail -1) $(date +%T)" >> runs/vm2_q224.log
