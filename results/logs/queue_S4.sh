#!/bin/bash
# 448x448 track: record the seed-42 views at 448, upscale the final 224 set (SZH2) as the start, then one full
# lineage (exact PFD 300 it -> 2 rounds -> hybrid -> pooled scores) with 4096 train rows held out.
cd /home/ubuntu/Uriel/data_centric/work
PY=/home/ubuntu/vllm-env/bin/python
export DD_HOLDOUT=4096
[ -f cache/views_n10_s42_sz448.pt ] || $PY views_for_n.py 10 42 448 > runs/views_sz448.log 2>&1
$PY - <<'PYEOF'
import torch, torch.nn.functional as F
d = torch.load('cache/sm_SZH2_best.pt')
x = F.interpolate(d['x01'], size=448, mode='bicubic', align_corners=False).clamp(0, 1).mul(255).round().div(255)
torch.save({'x01': x, 'y': d['y']}, 'cache/init_SZH2_sz448.pt'); print(tuple(x.shape))
PYEOF
runs/lineage2.sh cache/init_SZH2_sz448.pt cache/views_n10_s42_sz448.pt S4 300
