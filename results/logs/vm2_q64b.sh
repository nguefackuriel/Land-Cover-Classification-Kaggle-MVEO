#!/bin/bash
# VM2, 64 track (2/11): new 64 lineage from init_c4j0 (the medoids behind C2J0, our best 224 lineage, never run at 64),
# then dithered gradient-weighted rounds 2-5. Logs runs/lineage_Q4J0.log, runs/dither_Q4J0G.log.
cd /home/ubuntu/Uriel/data_centric/work
export DD_HOLDOUT=4096
runs/lineage2.sh cache/init_c4j0.pt cache/views_n10_s42.pt Q4J0 400
V=cache/views_n10_s42.pt runs/queue_dither.sh cache/sm_Q4J0H1_best.pt Q4J0G 2 5 0 --weight grad
