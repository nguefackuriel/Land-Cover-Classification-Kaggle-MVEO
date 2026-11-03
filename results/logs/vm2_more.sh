#!/bin/bash
# Continue the two 224 lineages that still climb: KS (step-w, 1000-it targets; last-4 val 87.67 -> 87.87 -> 87.93)
# rounds 4-6 after vm2_ks.sh, and NS (step-w, 3000-it targets) rounds 3-4 after vm2_ns.sh. Each starts from the best
# pooled set of the previous round (from its dither log).
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
last() { grep "done best" runs/dither_$1.log | tail -1 | awk '{print $5}'; }
( while kill -0 37408 2> /dev/null; do sleep 60; done
  runs/queue_dither.sh $(last KS) KS 4 6 0 $KN ) &
( while kill -0 39472 2> /dev/null; do sleep 60; done
  NIT=3000 runs/queue_dither_n.sh $(last NS) NS 3 4 0 $KN ) &
wait
echo "more done $(date +%T)" >> runs/vm2_q224.log
