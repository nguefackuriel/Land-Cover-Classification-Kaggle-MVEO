"""Shuffle the rows of a set (same images and labels, new file order). Under the fixed seed-42 loader every image then
gets other crops/flips in all 5 epochs: a basin hop for the n=10 plateau, the n=10 twin of the n=11 restart.
usage: perm_set.py <set.pt> <out.pt> <seed>"""
import sys, torch
import dd_lib as D
x, y = D.load_set(sys.argv[1])
p = torch.randperm(len(y), generator=torch.Generator().manual_seed(int(sys.argv[3])))
torch.save({'x01': x[p], 'y': y[p], 'it': 0}, sys.argv[2])
print(sys.argv[2], 'order', p.tolist())
