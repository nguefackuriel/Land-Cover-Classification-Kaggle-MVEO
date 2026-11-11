#!/bin/bash
# Size curve, second try (11/11). On VM3 the three sizes ran side by side and ran out of GPU memory: 112 rounds 1-2,
# all of 160 and 224 rounds 2-3 crashed in signmatch. Here, one job at a time: 224 rounds 2-3 from the only good 224
# round (sm_SC224r1_it300, tag SC224b), then 112 rounds 2-3 from its only good round (sm_SC112r3_it280 = effectively
# round 1, tag SC112b). 160 runs alone on VM3 (runs/vm3_curve2.sh).
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True IT=300 NIT=3000
KN="--weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014"
echo "curve2 start $(date +%T)" >> runs/vm2_curve.log
V=cache/views_n10_s42_sz224.pt runs/queue_dither_n.sh cache/sm_SC224r1_it300.pt SC224b 2 3 0 $KN
V=cache/views_n10_s42_sz112.pt runs/queue_dither_n.sh cache/sm_SC112r3_it280.pt SC112b 2 3 0 $KN
echo "curve2 done $(date +%T)" >> runs/vm2_curve.log
