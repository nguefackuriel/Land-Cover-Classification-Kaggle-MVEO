#!/bin/bash
# KS: 224 rounds with step weights (KTa: same peak as KT2, 87.70, but last-4 val 87.67 instead of 87.49).
# Waits for NLK (vm2_q224c.sh), starts from the run with the higher last-4 mean val (KTa or NLK), at its best pooled
# checkpoint, then runs queue_dither rounds 1-3 with all knobs + step weights. Logs runs/vm2_q224.log, runs/dither_KS.log.
cd /home/ubuntu/Uriel/data_centric/work
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
L=runs/vm2_q224.log
while kill -0 17865 2> /dev/null; do sleep 60; done
read S R <<< "$(python3 -c "
import json
def load(t): return [json.loads(l) for l in open(f'runs/pooled_{t}.log')]
def m(t): r=load(t)[-4:]; return sum(d['val'] for d in r)/len(r)
t=max(['KTa','NLK'],key=m); b=max(load(t),key=lambda d:d['pooled'])
print(b['set'], t+'_last4=%.4f' % m(t), 'KTa=%.4f NLK=%.4f' % (m('KTa'), m('NLK')))")"
echo "KS start $S ($R) $(date +%T)" >> $L
runs/queue_dither.sh $S KS 1 3 0 --weight grad --tau 0.15 --pfd-w 5.0 --real-bs 2048 --step-w 1,0.462,0.205,0.071,0.014
echo "KS done $(date +%T)" >> $L
