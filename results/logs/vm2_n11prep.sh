#!/bin/bash
# Prep for the n=11 test and the pixel-soup test: official views for 11 images (224 and 64), per-class val accuracy of the
# best sets, and exact scores of averaged sets (same lineage).
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1
L=runs/vm2_n11.log; V2=cache/views_n10_s42_sz224.pt; V6=cache/views_n10_s42.pt
echo "prep start $(date +%T)" >> $L
$PY views_for_n.py 11 42 224 >> $L 2>&1
$PY views_for_n.py 11 42 >> $L 2>&1
$PY views_confusion.py cache/sm_NL1_it300.pt $V2 > runs/conf_NL1i300.log 2>&1
$PY views_confusion.py cache/sm_A64R3_it200.pt $V6 > runs/conf_A64R3i200.log 2>&1
$PY soup_set.py cache/soup_NL4.pt cache/sm_NL1_it300.pt cache/sm_NL2_it400.pt cache/sm_NLs1_it300.pt cache/sm_NL2x_it600.pt >> $L 2>&1
$PY soup_set.py cache/soup_NL2.pt cache/sm_NL1_it300.pt cache/sm_NLs1_it300.pt >> $L 2>&1
$PY soup_set.py cache/soup_NLs1t.pt cache/sm_NLs1_it260.pt cache/sm_NLs1_it280.pt cache/sm_NLs1_it300.pt >> $L 2>&1
$PY soup_set.py cache/soup_A4.pt cache/sm_A64R3_it200.pt cache/sm_A64R4_it260.pt cache/sm_A64R5_it300.pt cache/sm_A64R6_it260.pt >> $L 2>&1
$PY soup_set.py cache/soup_A3t.pt cache/sm_A64R3_it160.pt cache/sm_A64R3_it180.pt cache/sm_A64R3_it200.pt >> $L 2>&1
$PY exact_eval.py $V2 cache/sm_NL1_it300.pt cache/soup_NL4.pt cache/soup_NL2.pt cache/soup_NLs1t.pt 2>/dev/null | grep '^{' > runs/pooled_soup224.log
$PY exact_eval.py $V6 cache/sm_A64R3_it200.pt cache/soup_A4.pt cache/soup_A3t.pt 2>/dev/null | grep '^{' > runs/pooled_soup64.log
echo "prep done $(date +%T)" >> $L
