"""Average the pixels of several sets with the same labels (same lineage, close in pixel space) and round back to
8 bits, like averaging the weights of fine-tuned models. usage: soup_set.py <out.pt> <set1.pt> <set2.pt> [...]"""
import sys, torch
import dd_lib as D

xs, y0 = [], None
for p in sys.argv[2:]:
    x, y = D.load_set(p)
    assert y0 is None or torch.equal(y, y0), p
    xs.append(x.float()); y0 = y
x = torch.stack(xs).mean(0).mul(255).round().div(255)
torch.save({'x01': x, 'y': y0, 'it': -1}, sys.argv[1])
print(sys.argv[1], 'from', len(xs), 'sets, mean |change| vs first:', round((x - xs[0]).abs().mean().item() * 255, 3), 'levels')
