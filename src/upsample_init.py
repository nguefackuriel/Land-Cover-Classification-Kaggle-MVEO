"""Upsample a distilled set (x01 at 64x64) to size S with PIL bilinear (as a viewer would), keep labels; optional index
subset. usage: upsample_init.py <src.pt|dir> <S> <out.pt> [i,j,k]"""
import sys, numpy as np, torch
from PIL import Image
import dd_lib as D
x, y = D.load_set(sys.argv[1]); S = int(sys.argv[2])
if len(sys.argv) > 4:
    idx = [int(i) for i in sys.argv[4].split(',')]; x, y = x[idx], y[idx]
u8 = (x * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
up = np.stack([np.asarray(Image.fromarray(im, 'RGB').resize((S, S), Image.BILINEAR)) for im in u8])
torch.save({'x01': torch.from_numpy(up).permute(0, 3, 1, 2).float().div(255), 'y': y}, sys.argv[3])
print(sys.argv[3], tuple(up.shape), y.tolist())
