"""Same-class g x g mosaics of real images (RDED-style): each 64x64 image packs g^2 k-means medoids downscaled to
64/g, so a batch of 32 carries 32 g^2 real images. Proto accuracy over 3 seeds, vs. plain medoids at the same n.
usage: mosaic_eval.py 120 64 40      (n values; 10 classes x n/10 mosaics)"""
import json, sys, torch, torch.nn.functional as F
import dd_lib as D

G = 2
f = D.features()
for n in map(int, sys.argv[1:]):
    k = [n // 10] * 10
    idx = D.kmeans_medoids(f, [c * G * G for c in k])
    x, y = D.as_x01(idx)
    tiles = F.interpolate(x, size=64 // G, mode='area')
    xs, ys, gen = [], [], torch.Generator().manual_seed(0)
    for c in range(10):
        ids = (y == c).nonzero().squeeze(1)
        ids = ids[torch.randperm(len(ids), generator=gen)]
        for j in range(k[c]):
            t = tiles[ids[j * G * G:(j + 1) * G * G]]                 # (G*G, 3, 64/G, 64/G)
            rows = [torch.cat(list(t[r * G:(r + 1) * G]), dim=2) for r in range(G)]
            xs.append(torch.cat(rows, dim=1)); ys.append(c)
    xm, ym = torch.stack(xs), torch.tensor(ys)
    torch.save({'x01': (xm * 255).round() / 255, 'y': ym}, f'cache/mosaic{G}_n{n}.pt')
    accs = [round(D.train_eval((xm * 255).round() / 255, ym, s), 4) for s in (42, 0, 1)]
    mean = sum(accs) / 3
    print(json.dumps({'n': n, 'grid': G, 'real_images': len(idx), 'accs': accs, 'mean': round(mean, 4),
                      'score_mean': round(D.score(mean, n), 4)}), flush=True)
