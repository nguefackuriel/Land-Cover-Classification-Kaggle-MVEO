"""Downsample a set (x01 in [0,1], N x 3 x H x W) to S x S with PIL (box or bicubic, antialiased), keep labels; 8-bit.
usage: down_set.py <set.pt> <out.pt> <size> [box|bicubic|bilinear|lanczos]"""
import sys, numpy as np, torch
from PIL import Image
src, out, S = sys.argv[1], sys.argv[2], int(sys.argv[3])
mode = {'box': Image.BOX, 'bicubic': Image.BICUBIC, 'bilinear': Image.BILINEAR, 'lanczos': Image.LANCZOS}[sys.argv[4] if len(sys.argv) > 4 else 'box']
d = torch.load(src, map_location='cpu')
x = (d['x01'].float().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
y = [np.asarray(Image.fromarray(a).resize((S, S), mode)) for a in x]
x2 = torch.from_numpy(np.stack(y)).permute(0, 3, 1, 2).float() / 255
torch.save({'x01': x2, 'y': d['y'], 'it': -1}, out)
print(src, '->', out, tuple(x2.shape))
