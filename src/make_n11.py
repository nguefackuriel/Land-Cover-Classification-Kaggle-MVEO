"""Add one image of class c to a distilled set (n=10 -> 11). mode copy: a copy of the set's own class-c image (the two
copies get different official crops/flips, so they part ways under distillation); mode medoid: a real class-c medoid
(not in the held-out rows), resized with PIL bilinear to the set's size.
With the two views files, the rows are put in the order that gives every old image its old epoch-0 view: the official
seed-42 loader draws the same crop/flip for batch position p in epoch 0 whatever n is (one batch, one worker), only the
shuffle differs; epochs 1-4 differ anyway. Without this order the set starts at ~24% (88% at n=10).
usage: DD_HOLDOUT=4096 make_n11.py <set.pt> <class> <copy|medoid> <out.pt> [<views n10> <views n11>]"""
import sys, numpy as np, torch
from PIL import Image
import dd_lib as D

x, y = D.load_set(sys.argv[1]); c, mode = int(sys.argv[2]), sys.argv[3]
S = x.shape[-1]
if mode == 'copy':
    i = (y == c).nonzero()[0, 0]
    xn = x[i:i + 1].clone()
else:
    hold = set(D.HOLD.tolist()) if len(D.HOLD) else set()
    rows = [r for r in torch.load('cache/medoids_ipc4.pt').tolist()
            if D.ytr[r] == c and r not in hold]
    xm, _ = D.as_x01(torch.tensor(rows[:1]))
    u8 = (xm[0] * 255).round().to(torch.uint8).permute(1, 2, 0).numpy()
    im = Image.fromarray(u8, 'RGB')
    if S != im.size[0]:
        im = im.resize((S, S), Image.BILINEAR)
    xn = torch.from_numpy(np.asarray(im)).permute(2, 0, 1)[None].float().div(255)
x2, y2 = torch.cat([x, xn]), torch.cat([y, torch.tensor([c])])
if len(sys.argv) > 6:
    a0 = torch.load(sys.argv[5])['epochs'][0][0][0].tolist()    # epoch 0, batch 0: row index at each batch position
    b0 = torch.load(sys.argv[6])['epochs'][0][0][0].tolist()
    order = [None] * 11
    for p in range(10):
        order[b0[p]] = a0[p]      # new row b0[p] sits at position p in epoch 0, like old row a0[p] did
    order[b0[10]] = 10            # the added image takes the new last position
    x2, y2 = x2[order], y2[order]
    print('row order', order)
torch.save({'x01': x2, 'y': y2, 'it': 0}, sys.argv[4])
print(sys.argv[4], tuple(x2.shape), y2.tolist())
