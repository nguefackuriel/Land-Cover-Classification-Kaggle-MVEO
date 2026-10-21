"""New lineage starts: k medoids per class (k-means seed s) from the distillation pool only, so with DD_HOLDOUT set
no held-out row is used. Start j takes the j-th medoid of every class. Writes cache/init_c{k}s{s}j{j}.pt (64x64) and
the bicubic 224 version. usage: DD_HOLDOUT=4096 python make_inits.py <k> <seed>"""
import sys, torch, torch.nn.functional as F
import dd_lib as D

k, s = int(sys.argv[1]), int(sys.argv[2])
sel = D.kmeans_medoids(D.features(), [k] * 10, seed=s, rows=D.POOL).view(10, k)
assert not torch.isin(sel, D.HOLD).any()
for j in range(k):
    x, y = D.as_x01(sel[:, j])
    torch.save({'x01': x, 'y': y}, f'cache/init_c{k}s{s}j{j}.pt')
    x = F.interpolate(x, size=224, mode='bicubic', align_corners=False).clamp(0, 1).mul(255).round().div(255)
    torch.save({'x01': x, 'y': y}, f'cache/init_c{k}s{s}j{j}_sz224.pt')
    print(j, sel[:, j].tolist())
