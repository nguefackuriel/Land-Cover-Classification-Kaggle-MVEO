"""Per-class allocation at fixed n: k-means medoids for each allocation, proto accuracy over 6 seeds.
Classes: 0 AnnualCrop 1 Forest 2 Herbaceous 3 Highway 4 Industrial 5 Pasture 6 PermCrop 7 Residential 8 River 9 SeaLake"""
import json, sys, torch
import dd_lib as D

ALLOCS = {                      # A  F  H  Hw I  P  Pc R  Rv S
    'uniform12':               [12, 12, 12, 12, 12, 12, 12, 12, 12, 12],
    'mild':                    [12, 11, 12, 14, 10, 13, 13, 10, 14, 11],
    'strong':                  [12, 9, 12, 16, 9, 14, 14, 9, 16, 9],
}
SEEDS = (42, 0, 1, 2, 3, 4)
f = D.features()
for name, k in ALLOCS.items():
    assert sum(k) == 120, name
    idx = D.kmeans_medoids(f, k)
    torch.save(idx, f'cache/medoids_n120_{name}.pt')
    x, y = D.as_x01(idx)
    accs = [round(D.train_eval(x, y, s), 4) for s in SEEDS]
    print(json.dumps({'alloc': name, 'k': k, 'accs': accs, 'mean': round(sum(accs) / len(accs), 4)}), flush=True)
