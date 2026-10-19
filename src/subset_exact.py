"""Exact seed-42 official accuracy (replayed views) of subsets of a distilled set: greedy backward elimination.
usage: PROTO_PIL=1 subset_exact.py <set.pt> <n_min>"""
import json, sys, torch
import dd_lib as D
src, n_min = sys.argv[1], int(sys.argv[2])
x, y = D.load_set(src)
keep = list(range(len(y)))
def acc(idx):
    V = torch.load(f'cache/views_n{len(idx)}_s42.pt')
    return D.train_eval_views(x[idx], y[idx], 42, V)[-1]
print(json.dumps({'n': len(keep), 'acc': round(acc(keep), 4), 'labels': y[keep].tolist()}), flush=True)
while len(keep) > n_min:
    res = []
    for r in keep:
        idx = [k for k in keep if k != r]
        res.append((acc(idx), r))
    res.sort(reverse=True)
    best, r = res[0]
    keep.remove(r)
    print(json.dumps({'n': len(keep), 'acc': round(best, 4), 'score': round(D.score(best, len(keep)), 4),
                      'removed': r, 'removed_label': int(y[r]), 'keep': keep,
                      'all': [(round(a, 4), int(y[i])) for a, i in res]}), flush=True)
