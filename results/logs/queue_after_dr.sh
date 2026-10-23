#!/bin/bash
# After the DR plateau (DR3 it160, pooled 86.72): new seeds to leave the fixed point of the deterministic round map,
# a longer round, then dithered rounds on the best 64x64 lineage. Logs runs/dither_<tag>.log.
cd /home/ubuntu/Uriel/data_centric/work
NS=1 runs/queue_dither.sh cache/sm_DR3_it160.pt DS 1 2 0 --dither-seed 1
NS=2 IT=400 runs/queue_dither.sh cache/sm_DR3_it160.pt DL 1 1 0 --dither-seed 2
V=cache/views_n10_s42.pt runs/queue_dither.sh cache/sm_P8J3H4_it160.pt D64_ 1 3 0
