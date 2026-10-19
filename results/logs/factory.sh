#!/bin/bash
# Lineages one after the other (one GPU job at a time, the QA2026 vLLM job shares the GPU).
cd /home/ubuntu/Uriel/data_centric/work
while pgrep -f "tag [h]ybw5" > /dev/null || pgrep -f "queue_after_n[9]" > /dev/null; do sleep 60; done
# the medoid lineage already has its PFD stage and one round: round 2, then the hybrid
runs/loop_rounds2.sh cache/pfd3_pf_MEDP1_best.pt cache/views_n10_s42.pt MEDP 2 2
runs/loop_hybrid.sh cache/pfd3_pf_MEDP2_best.pt MEDPH 1 1
for j in 0 1 2 3; do runs/lineage.sh cache/init_c4j$j.pt C4J$j; done
