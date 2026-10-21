"""Shared helpers: ViT features of the train set, coreset selectors, fast multi-seed scoring with proto."""
import math, os, sys, torch, torch.nn.functional as F
from pathlib import Path
import proto as P

xtr, ytr = P.load_split('train'); xva, yva = P.load_split('validation')
FEAT = Path('cache/train_feats.pt')

# DD_HOLDOUT=k keeps the first k rows of a fixed train permutation (seed 33, the order exact_eval.py uses) out of
# everything distillation samples, so val + HOLD is a clean selection set. Unset: POOL is all rows, same draws as before.
_HK = int(os.environ.get('DD_HOLDOUT', '0'))
_PERM = torch.randperm(len(xtr), generator=torch.Generator().manual_seed(33))
HOLD = _PERM[:_HK]
POOL = _PERM[_HK:].sort().values if _HK else torch.arange(len(xtr))

def sample_train(k, generator=None):
    """k random train row indices drawn from POOL."""
    return POOL[torch.randperm(len(POOL), generator=generator)[:k]]

@torch.no_grad()
def features():
    if FEAT.exists():
        return torch.load(FEAT)
    m, _, _ = P.fresh_model(0); m.heads = torch.nn.Identity(); m.eval(); out = []
    for i in range(0, len(xtr), 256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(m(P.to_eval(xtr[i:i + 256].cuda())).float().cpu())
    f = F.normalize(torch.cat(out), dim=1); torch.save(f, FEAT); return f

def train_eval(x01, y, seed):
    """Official recipe on in-memory images (float NCHW in [0,1], any n): batch 32, shuffle, last partial batch
    kept, 5 epochs, per-epoch cosine lr, best-epoch val accuracy."""
    m, opt, sch = P.fresh_model(seed); best = 0
    x01, y = x01.cuda(), y.cuda()
    for ep in range(5):
        perm = torch.randperm(len(y))
        for j in range(0, len(y), 32):
            b = perm[j:j + 32]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment(x01[b])), y[b]).backward(); opt.step()
        sch.step(); best = max(best, P.evaluate(m, xva, yva))
    return best

def train_eval_views(x01, y, seed, views, last_only=False):
    """train_eval with the exact official batches and views of one seed (record_views.py output); per-epoch val
    (last_only: only the last epoch is scored, the others are None)."""
    m, opt, sch = P.fresh_model(seed); accs = []
    x01, y = x01.cuda(), y.cuda()
    E = len(views['epochs'])
    for e, batches in enumerate(views['epochs']):
        for idx, vw in batches:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment_views(x01[idx.cuda()], vw)), y[idx.cuda()]).backward(); opt.step()
        sch.step(); accs.append(P.evaluate(m, xva, yva) if (not last_only or e == E - 1) else None)
    return accs

def load_set(path):
    """(x01, y) of a set given as a folder with train.csv and its PNGs (submissions/*), a {'x01', 'y'} .pt saved by
    pfd3.py, or a .pt of training-set row indices (inits/medoids_*.pt)."""
    p = Path(path)
    if p.is_dir():
        import csv
        import numpy as np
        from PIL import Image
        rows = list(csv.DictReader(open(p / 'train.csv')))
        img = p / 'train' if (p / 'train').is_dir() else p
        x = torch.from_numpy(np.stack([np.asarray(Image.open(img / r['filename']).convert('RGB')) for r in rows]))
        return x.permute(0, 3, 1, 2).float().div(255), torch.tensor([P.CLASSES.index(r['label']) for r in rows])
    d = torch.load(p)
    return (d['x01'], d['y']) if isinstance(d, dict) else as_x01(d)

def as_x01(idx):
    return xtr[idx].permute(0, 3, 1, 2).float().div(255), ytr[idx]

def kmeans_medoids(f, per_class, iters=50, seed=0, rows=None):
    """per_class: list of 10 counts. k-means (cosine) per class, return the real image nearest each centroid.
    rows: only these train rows are used (e.g. POOL, to keep the held-out rows out)."""
    g = torch.Generator().manual_seed(seed); sel = []
    for c in range(10):
        k = per_class[c]; ids = (ytr == c).nonzero().squeeze(1)
        if rows is not None:
            ids = ids[torch.isin(ids, rows)]
        fc = f[ids]
        cent = fc[torch.randperm(len(fc), generator=g)[:k]].clone()
        for _ in range(iters):
            a = (fc @ cent.T).argmax(1)
            cent = torch.stack([F.normalize(fc[a == j].mean(0), dim=0) if (a == j).any() else cent[j] for j in range(k)])
        sims = fc @ cent.T; chosen = []
        for j in range(k):                      # nearest unused image to each centroid
            for i in sims[:, j].argsort(descending=True).tolist():
                if ids[i].item() not in chosen:
                    chosen.append(ids[i].item()); break
        sel += chosen
    return torch.tensor(sel)

def score(acc, n):
    return 0.5 * acc + 0.5 * (1 - math.log(n) / math.log(18900))
