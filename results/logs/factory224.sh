#!/bin/bash
# 224 lineage factory: the four k-means medoid inits (c4j0..3, real 64x64 images) upscaled to 224, then one full
# lineage each (exact PFD 400 it -> 2 rounds -> hybrid -> pooled), 4096 train rows held out. Results in
# runs/lineage_C2J*.log. usage: runs/factory224.sh [first_j] [last_j]
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096
V=cache/views_n10_s42_sz224.pt
for j in $(seq ${1:-0} ${2:-3}); do
  [ -f cache/init_c4j${j}_sz224.pt ] || $PY -c "
import torch, torch.nn.functional as F
d = torch.load('cache/init_c4j$j.pt')
x = F.interpolate(d['x01'], size=224, mode='bicubic', align_corners=False).clamp(0, 1).mul(255).round().div(255)
torch.save({'x01': x, 'y': d['y']}, 'cache/init_c4j${j}_sz224.pt')"
  runs/lineage2.sh cache/init_c4j${j}_sz224.pt $V C2J$j 400
done
