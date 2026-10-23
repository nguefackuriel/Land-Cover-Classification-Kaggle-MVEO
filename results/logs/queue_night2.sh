#!/bin/bash
# KD2 (dither + --weight grad, from DR3 it160) raised val from 86.49 to 87.01 in one round. Chain that recipe: rounds
# DG1..DG4 from KD2 it200, each with a fresh near target. Then KD3 (3-target consensus) and KD5 (heavier PFD) from the
# best DG set, then the two new 224 lineages. Logs runs/night.log and runs/dither_DG.log.
cd /home/ubuntu/Uriel/data_centric/work
runs/queue_dither.sh cache/sm_KD2_it200.pt DG 1 4 0 --weight grad
echo "DG rounds done: $(grep 'done best' runs/dither_DG.log | tr '\n' ' ') $(date +%T)" >> runs/night.log
for j in 2 3; do
  runs/lineage2.sh cache/init_c8s1j${j}_sz224.pt cache/views_n10_s42_sz224.pt C8J$j 400
  echo "C8J$j lineage done $(grep '^{' runs/lineage_C8J$j.log | tail -1) $(date +%T)" >> runs/night.log
  runs/queue_dither.sh cache/sm_C8J${j}H1_best.pt C8J${j}G 2 5 0 --weight grad
  echo "C8J$j rounds done $(tail -2 runs/dither_C8J${j}G.log | head -1) $(date +%T)" >> runs/night.log
done
echo "night2 done $(date +%T)" >> runs/night.log
