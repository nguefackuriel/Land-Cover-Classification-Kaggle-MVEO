"""Pixel-level check of proto.augment_views (PROTO_PIL=1) against torchvision's PIL path (crop + bilinear resize to 224)
on the recorded views of a set. usage: PROTO_PIL=1 pil_fidelity_224.py <set_dir_with_pngs> <views.pt>"""
import sys, torch, numpy as np
from pathlib import Path
from PIL import Image
from torchvision.transforms import functional as TF
import proto as P, dd_lib as D
x, y = D.load_set(sys.argv[1]); V = torch.load(sys.argv[2])
MEAN, STD = P.MEAN, P.STD
tot = diff = 0; maxd = 0; per_size = {}
for e, batches in enumerate(V['epochs']):
    for idx, vw in batches:
        ours = P.augment_views(x[idx], vw) * STD + MEAN                 # back to [0, 1]
        for k in range(len(idx)):
            i, j, h, w, hf, vf = [int(t) for t in vw[k]]
            img = Image.fromarray((x[idx[k]] * 255).round().byte().permute(1, 2, 0).numpy(), 'RGB')
            ref = TF.resized_crop(img, i, j, h, w, [224, 224], interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
            if hf: ref = TF.hflip(ref)
            if vf: ref = TF.vflip(ref)
            r = torch.from_numpy(np.asarray(ref)).permute(2, 0, 1).float()
            o = (ours[k] * 255).round()
            d = (o - r).abs()
            tot += d.numel(); diff += (d > 0).sum().item(); maxd = max(maxd, d.max().item())
            key = (h, w); per_size.setdefault(key, [0, 0]); per_size[key][0] += (d > 0).sum().item(); per_size[key][1] += d.numel()
print({'pixels': tot, 'mismatch_frac': round(diff / tot, 4), 'max_abs_diff_levels': maxd})
print('worst crops (h, w): frac', sorted(((round(a / b, 3), k) for k, (a, b) in per_size.items()), reverse=True)[:6])
