#!/bin/bash
# NS: 224 rounds that combine the two things that helped: a 3000-it near target (NLK: val max 87.83, pooled 87.95)
# and step weights (KTa: last-4 val 87.67 vs 87.49). Official check of NLK it160 first (small job), then waits for the
# Q4J0G queue (last 64 round) to free its slot, then rounds NS1-2 from NLK it160. Logs runs/vm2_q224.log, runs/dither_NS.log.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_q224.log
$PY export_synth.py cache/sm_NLK_it160.pt sets/n10_NLKi160_224 > runs/vm2_official_NLKi160.log 2>&1
$PY /home/ubuntu/Uriel/data_centric/ext/challenge_2026_public/baseline.py train --data sets/n10_NLKi160_224 \
    --output runs/vm2_official_NLKi160 >> runs/vm2_official_NLKi160.log 2>&1
rm -f runs/vm2_official_NLKi160/best.pt
echo "official NLK it160: $(grep -i 'val' runs/vm2_official_NLKi160.log | tail -1) $(date +%T)" >> $L
while kill -0 23713 2> /dev/null; do sleep 60; done
echo "NS start cache/sm_NLK_it160.pt NIT=3000 $(date +%T)" >> $L
NIT=3000 runs/queue_dither_n.sh cache/sm_NLK_it160.pt NS 1 2 0 --weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 \
    --step-w 1,0.462,0.205,0.071,0.014
echo "NS done $(date +%T)" >> $L
