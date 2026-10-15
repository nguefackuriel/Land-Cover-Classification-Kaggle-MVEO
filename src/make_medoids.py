"""k-means medoid index sets for arbitrary n (CPU only), same allocation rule as coreset_full_batches.py.
usage: make_medoids.py 97 65 33     -> cache/medoids_n97.pt ..."""
import collections, sys, torch
import dd_lib as D

HARD_FIRST = [2, 3, 6, 5, 8, 0, 7, 4, 1, 9]
def alloc(n):
    k = [n // 10] * 10
    for c in HARD_FIRST[:n % 10]:
        k[c] += 1
    return k

f = D.features()
for n in map(int, sys.argv[1:]):
    idx = D.kmeans_medoids(f, alloc(n))
    torch.save(idx, f'cache/medoids_n{n}.pt')
    print(n, 'steps', 5 * ((n + 31) // 32), 'alloc', alloc(n), sorted(collections.Counter(D.ytr[idx].tolist()).items()))
