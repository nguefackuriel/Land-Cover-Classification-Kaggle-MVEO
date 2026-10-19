#!/bin/bash
# One full lineage on the exact seed-42 run for any views file (n=10, any image size): exact PFD from an init,
# two rounds of near target -> sign matching -> PFD, one hybrid round, pooled score of each stage.
# usage: lineage2.sh <init.pt> <views.pt> <prefix> [pfd_iters]
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
INIT=$1; V=$2; P=$3; IT=${4:-400}
echo "start $INIT $V $(date +%T)" >> runs/lineage_$P.log
if [ ! -f cache/pfd3_${P}_pfd_best.pt ]; then
  PROTO_PIL=1 PROTO_EXACT=1 $PY pfd3.py --init $INIT --offload --heads 1 --lr 0.05 --lr-end 0.005 --iters $IT --eval-every 20 \
      --outer-T 0 --fixed-head-seed 42 --views $V --views-prob 1.0 --select-exact --fp32-inner --no-seed-eval \
      --tag ${P}_pfd > runs/pfd3_${P}_pfd.out 2>&1
fi
runs/loop_rounds2.sh cache/pfd3_${P}_pfd_best.pt $V ${P}R 1 2
runs/loop_hybrid2.sh cache/pfd3_pf_${P}R2_best.pt $V ${P}H 1 1
PROTO_PIL=1 PROTO_EXACT=1 $PY exact_eval.py $V cache/pfd3_${P}_pfd_best.pt cache/pfd3_pf_${P}R1_best.pt cache/pfd3_pf_${P}R2_best.pt \
    cache/sm_${P}H1_best.pt 2>/dev/null | grep '^{' >> runs/lineage_$P.log
echo "done $(date +%T)" >> runs/lineage_$P.log
