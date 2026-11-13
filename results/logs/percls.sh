#!/bin/bash
# 13/11: per-class recall of the 224 start sets (raw), their round-3 bests and the lineage ends. Log runs/percls.log.
cd /home/ubuntu/Uriel/data_centric/work
export PROTO_PIL=1 PROTO_EXACT=1 DD_HOLDOUT=4096
C=cache
/home/ubuntu/vllm-env/bin/python percls_eval.py $C/views_n10_s42_sz224.pt \
  $C/sm_SC224b3_it300.pt $C/sm_F224C83_it200.pt $C/sm_F224C23_it280.pt $C/sm_S224J23_it260.pt $C/sm_S224J33_it300.pt \
  $C/sm_S224J43_it260.pt $C/sm_S224J53_it300.pt $C/sm_S224J63_it260.pt $C/sm_S224J73_it300.pt \
  $C/sm_SC224b11_it260.pt $C/sm_NL1_it300.pt $C/sm_F224C811_it260.pt $C/sm_F224C211_it300.pt $C/sm_S224J68_it280.pt \
  $C/sm_S224K58_it300.pt \
  $C/init_c4j0_sz224.pt $C/init_c4j1_sz224.pt $C/init_c4j2_sz224.pt $C/init_c4j3_sz224.pt \
  $C/init_c8s1j0_sz224.pt $C/init_c8s1j1_sz224.pt $C/init_c8s1j2_sz224.pt $C/init_c8s1j3_sz224.pt \
  $C/init_c8s1j4_sz224.pt $C/init_c8s1j5_sz224.pt $C/init_c8s1j6_sz224.pt $C/init_c8s1j7_sz224.pt > runs/percls.log 2> runs/percls.err
echo "percls done $(date +%T)" >> runs/percls.log
