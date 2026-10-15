"""Budgets that are exact multiples of 32 (no small leftover batch). Extra slots go to the classes usually
confused in EuroSAT (Herbaceous, Highway, PermanentCrop, Pasture, River, AnnualCrop, ...)."""
import sys, torch
import dd_lib as D

HARD_FIRST = [2, 3, 6, 5, 8, 0, 7, 4, 1, 9]
def alloc(n):
    k = [n // 10] * 10
    for c in HARD_FIRST[:n % 10]:
        k[c] += 1
    return k

f = D.features()
for n in [64, 96, 128, 160]:
    idx = D.kmeans_medoids(f, alloc(n)); x, y = D.as_x01(idx)
    accs = [round(D.train_eval(x, y, s), 4) for s in [42, 0, 1]]
    mean = sum(accs) / 3
    print(f'n={n} steps={5*((n+31)//32)} alloc={alloc(n)} accs={accs} mean={mean:.4f} '
          f'score@42={D.score(accs[0], n):.4f} score@mean={D.score(mean, n):.4f}', flush=True)
    torch.save(idx, f'cache/medoids_n{n}.pt')
