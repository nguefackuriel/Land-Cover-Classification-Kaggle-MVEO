"""k-means medoids with other k-means seeds (alternative PFD inits).
usage: make_medoids_km.py 1 2              -> uniform 12/class, cache/medoids_n120_km{s}.pt
       make_medoids_km.py --n 112 1 2      -> make_medoids.alloc(112), cache/medoids_n112_km{s}.pt"""
import sys, torch
import dd_lib as D
HARD_FIRST = [2, 3, 6, 5, 8, 0, 7, 4, 1, 9]      # same rule as make_medoids.alloc
args = sys.argv[1:]
n = int(args[1]) if args[0] == '--n' else 120
seeds = list(map(int, args[2:] if args[0] == '--n' else args))
k = [n // 10] * 10
for c in HARD_FIRST[:n % 10]:
    k[c] += 1
ref = torch.load('cache/medoids_ipc12.pt' if n == 120 else f'cache/medoids_n{n}.pt')
f = D.features()
for s in seeds:
    idx = D.kmeans_medoids(f, k, seed=s)
    torch.save(idx, f'cache/medoids_n{n}_km{s}.pt')
    print(s, len(idx), len(set(idx.tolist()) & set(ref.tolist())), 'shared with km0')
