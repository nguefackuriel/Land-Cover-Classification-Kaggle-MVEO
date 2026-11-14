#!/bin/bash
# 14/11 03:00: stop the HYA lineage (c4j0 + c8s1j6 Highway/River: low tier, pooled r3 85.34), then one round each for more
# hybrids on the c8s1j6 base: HYC (+ c4j0 PermCrop), HYD (+ c4j0 AnnualCrop), HYE (+ c4j0 AnnualCrop, Forest, PermCrop,
# Residential). Round 1 to compare with HYB (j6 + c4j0 AnnualCrop + PermCrop) 84.97, j6 84.56, c4j0 84.02.
cd /home/ubuntu/Uriel/data_centric/work
T=HY; T=${T}A
pkill -f "cont224.sh [^ ]* $T "; pkill -f "queue_dither_n.sh [^ ]* $T "; sleep 2
for s in TERM KILL; do for p in $(pgrep -f "_$T"); do [ "$p" != $$ ] && kill -$s $p 2> /dev/null; done; sleep 5; done
echo "$T stopped $(date +%T)" >> runs/switch224.log
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
for h in C D E; do V=cache/views_n10_s42_sz224.pt runs/queue_dither_n.sh cache/init_hyb${h}_sz224.pt HY$h 1 1 0 $KN; done
echo "hyb r1 done $(date +%T)" >> runs/switch224.log
