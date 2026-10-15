"""Coreset baselines (random vs. feature-centroid medoids) scored with the fast protocol replica over several
training seeds, so selection quality is not confounded with the seed-42 head."""
import sys, json, torch, numpy as np, torch.nn.functional as F
import proto as P

xtr, ytr = P.load_split('train'); xva, yva = P.load_split('validation')

@torch.no_grad()
def features():
    m, _, _ = P.fresh_model(0); m.heads = torch.nn.Identity(); m.eval(); out = []
    for i in range(0, len(xtr), 256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(m(P.to_eval(xtr[i:i + 256].cuda())).float().cpu())
    return F.normalize(torch.cat(out), dim=1)

def train_eval(idx, seed):
    """Official recipe for n<=32: one batch per epoch, 5 epochs, best-epoch val selection."""
    m, opt, sch = P.fresh_model(seed); best = 0
    x, y = xtr[idx].cuda().permute(0, 3, 1, 2).float().div(255), ytr[idx].cuda()
    for ep in range(5):
        perm = torch.randperm(len(idx))
        for j in range(0, len(idx), 32):
            b = perm[j:j + 32]
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment(x[b])), y[b]).backward(); opt.step()
        sch.step(); best = max(best, P.evaluate(m, xva, yva))
    return best

def kmeans_medoids(f, ipc, iters=50):
    sel = []
    for c in range(10):
        ids = (ytr == c).nonzero().squeeze(1); fc = f[ids]
        cent = fc[torch.randperm(len(fc))[:ipc]].clone()
        for _ in range(iters):
            a = (fc @ cent.T).argmax(1)
            cent = torch.stack([F.normalize(fc[a == k].mean(0), dim=0) if (a == k).any() else cent[k] for k in range(ipc)])
        sel += [ids[(fc @ cent[k]).argmax()].item() for k in range(ipc)]
    return torch.tensor(sel)

f = features()
for ipc in [1, 2, 3]:
    res = {}
    g = torch.Generator().manual_seed(0)
    rnd = torch.cat([(ytr == c).nonzero().squeeze(1)[torch.randperm((ytr == c).sum().item(), generator=g)[:ipc]] for c in range(10)])
    for name, idx in [('random', rnd), ('kmeans_medoid', kmeans_medoids(f, ipc))]:
        res[name] = [round(train_eval(idx, s), 4) for s in [42, 0, 1]]
    print(f'ipc={ipc} n={10*ipc}', json.dumps(res), flush=True)
