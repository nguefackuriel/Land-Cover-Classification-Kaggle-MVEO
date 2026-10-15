"""Is the meta-optimisation overfitting the training images used in its outer loss? Train on a set (official recipe,
replica) and report accuracy on validation and on a random 4 050-image subset of train (eval transform), per seed.
usage: train_val_gap.py cache/a.pt cache/b.pt ...      (index files or {'x01','y'} dicts)"""
import json, sys, torch
import dd_lib as D
import proto as P

g = torch.Generator().manual_seed(123)
tidx = torch.randperm(len(D.xtr), generator=g)[:len(D.xva)]
xt, yt = D.xtr[tidx], D.ytr[tidx]


def train(x01, y, seed):
    m, opt, sch = P.fresh_model(seed)
    x01, y = x01.cuda(), y.cuda()
    for ep in range(5):
        perm = torch.randperm(len(y))
        for j in range(0, len(y), 32):
            b = perm[j:j + 32]
            opt.zero_grad(set_to_none=True)
            torch.nn.functional.cross_entropy(m(P.augment(x01[b])), y[b]).backward(); opt.step()
        sch.step()
    return m


for f in sys.argv[1:]:
    x, y = D.load_set(f)
    rows = []
    for s in (42, 0, 1):
        m = train(x, y, s)
        rows.append((P.evaluate(m, D.xva, D.yva), P.evaluate(m, xt, yt)))
        del m
    va, tr = [sum(r[i] for r in rows) / len(rows) for i in (0, 1)]
    print(json.dumps({'set': f, 'val': round(va, 4), 'train_subset': round(tr, 4), 'gap': round(tr - va, 4),
                      'per_seed': [[round(a, 4), round(b, 4)] for a, b in rows]}), flush=True)
