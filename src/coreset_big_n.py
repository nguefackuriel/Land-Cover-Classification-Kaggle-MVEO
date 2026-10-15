"""Coreset frontier at larger budgets (10-20 official steps): k-means medoids, uniform per class."""
import json, sys
import dd_lib as D

f = D.features()
for ipc in [4, 6, 10, 12]:
    idx = D.kmeans_medoids(f, [ipc] * 10)
    x, y = D.as_x01(idx)
    accs = [round(D.train_eval(x, y, s), 4) for s in [42, 0, 1]]
    n = len(idx); mean = sum(accs) / len(accs)
    print(f'ipc={ipc} n={n} steps={5*((n+31)//32)} accs={accs} mean={mean:.4f} '
          f'score@42={D.score(accs[0], n):.4f} score@mean={D.score(mean, n):.4f}', flush=True)
    D.torch.save(idx, f'cache/medoids_ipc{ipc}.pt')
