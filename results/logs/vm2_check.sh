#!/bin/bash
# VM2 sanity: exact replay must match the main machine (KD2 224: val 0.8701 hold 0.8713; D64_1_it180: val 0.843 hold ~0.852),
# official baseline.py on the KD2 PNGs must give val 87.01 with the same Kaggle predictions, then 224->64 projections.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096 PROTO_PIL=1
PROTO_EXACT=1 $PY exact_eval.py cache/views_n10_s42_sz224.pt cache/sm_KD2_it200.pt 2>/dev/null | grep "^{" > runs/vm2_check.log
PROTO_EXACT=1 $PY exact_eval.py cache/views_n10_s42.pt cache/sm_D64_1_it180.pt 2>/dev/null | grep "^{" >> runs/vm2_check.log
$PY proj64.py cache/sm_KD2_it200.pt cache/views_n10_s42.pt cache/views_n10_s42_sz224.pt cache/proj64_KD2 >> runs/vm2_check.log 2>&1
PROTO_EXACT=1 $PY exact_eval.py cache/views_n10_s42.pt cache/proj64_KD2_box.pt cache/proj64_KD2_bil.pt cache/proj64_KD2_ls.pt 2>/dev/null | grep "^{" >> runs/vm2_check.log
$PY /home/ubuntu/Uriel/data_centric/ext/challenge_2026_public/baseline.py train --data submissions/n10_KD2_224_images --output runs/vm2_official_KD2 > runs/vm2_official_KD2.log 2>&1
echo "done $(date +%T)" >> runs/vm2_check.log
