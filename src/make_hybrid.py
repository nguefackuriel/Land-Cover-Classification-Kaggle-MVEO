"""Hybrid start set: the rows of a base start with some class images taken from a donor start (rows are ordered by class).
usage: make_hybrid.py <base.pt> <donor.pt> <out.pt> <class> [<class> ...]"""
import sys, torch
b, d = torch.load(sys.argv[1]), torch.load(sys.argv[2])
assert torch.equal(b['y'], d['y'])
x = b['x01'].clone()
for c in map(int, sys.argv[4:]):
    r = (b['y'] == c).nonzero().item()
    x[r] = d['x01'][r]
torch.save({'x01': x, 'y': b['y'].clone()}, sys.argv[3])
print(sys.argv[3], 'classes from donor:', sys.argv[4:])
