#!/bin/bash
# 12/11: does the 224 ceiling depend on the starting images or on the history of the lineage? The final recipe from
# the c4j0 medoids reached 88.57 (SC224b, round 11), like NL1 (also c4j0, older recipes first). L224C8 / L224C2 came
# from other medoids with older-recipe rounds first and stopped at 87.17 / 86.51 after 10 final-recipe rounds.
# Here: the final recipe from scratch on other medoids, 11 rounds. usage: fresh224.sh <init> <tag> <pid to wait for>
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
V=cache/views_n10_s42_sz224.pt runs/queue_dither_n.sh $1 $2 1 11 $3 $KN
