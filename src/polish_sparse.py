"""Greedy uint8 polish of a set on the exact seed-42 run: each step, exact meta-gradient (pfd3 machinery, fixed head and
views), then +-1 level on the top-k pixels by |g| (k in --ks, along -sign g), each candidate scored by the exact replay:
calibrated CE on --ntrain fixed training images (not validation). Keep the best candidate if it lowers the CE.
usage: PROTO_PIL=1 polish_sparse.py --init set.pt --views views.pt --iters 100 --tag ps1"""
import argparse, json, sys, time, torch, torch.nn.functional as F
from torch.func import functional_call
ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--views', required=True)
ap.add_argument('--iters', type=int, default=100)
ap.add_argument('--ks', default='10,30,100')
ap.add_argument('--ntrain', type=int, default=4096)
ap.add_argument('--eval-every', type=int, default=10)
ap.add_argument('--tag', default='ps')
a = ap.parse_args()
sys.argv = ['pfd3.py', '--init', a.init, '--offload', '--fixed-head-seed', '42', '--views', a.views, '--fp32-inner', '--outer-T', '0']
import pfd3 as Q
D, P, dev = Q.D, Q.P, Q.dev
Q.USE_VIEWS[0] = True
head = Q.sample_head()
x = (Q.pixels(Q.z.detach()) * 255).round() / 255
y = Q.y
RIDX = torch.randperm(len(D.xtr), generator=torch.Generator().manual_seed(21))[:a.ntrain]
KS = [int(k) for k in a.ks.split(',')]


@torch.no_grad()
def logits(th, xu8):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return torch.cat([functional_call(Q.base, th, (P.to_eval(xu8[j:j + 256].to(dev)),)).float() for j in range(0, len(xu8), 256)])


def objective(xc, val=False):
    w = Q.a.window; Q.a.window = 0
    th, _, _ = Q.forward(xc, head)
    Q.a.window = w
    o = logits(th, D.xtr[RIDX]); yb = D.ytr[RIDX].to(dev)
    ce = min(F.cross_entropy(o / T, yb).item() for T in torch.logspace(-2, 1, 61, device=dev))
    va = (logits(th, D.xva).argmax(1).cpu() == D.yva).float().mean().item() if val else None
    return ce, va


cur, va0 = objective(x, val=True)
log = open(f'runs/polish_{a.tag}.jsonl', 'w')
print(json.dumps({'it': 0, 'ce': round(cur, 5), 'val': round(va0, 4)}), flush=True)
best_val = va0
for it in range(1, a.iters + 1):
    t0 = time.time()
    thK, saved, K = Q.forward(x, head)
    G, rl = Q.real_grad(thK, RIDX[torch.randperm(a.ntrain)[:2048]])
    del thK
    gx = Q.backward(x.clone().requires_grad_(True), saved, G)
    del saved, G
    flat = gx.flatten(); order = flat.abs().argsort(descending=True)
    cands = []
    for k in KS:
        d = torch.zeros_like(flat); idx = order[:k]; d[idx] = -flat[idx].sign() / 255
        xc = (x + d.view_as(x)).clamp(0, 1)
        cands.append((objective(xc)[0], k, xc))
    ce, k, xc = min(cands, key=lambda c: c[0])
    acc = ce < cur
    if acc:
        x, cur = xc, ce
    rec = {'it': it, 'ce': round(cur, 5), 'tried': [(kk, round(c, 5)) for c, kk, _ in cands], 'accepted': k if acc else 0,
           'sec': round(time.time() - t0, 1)}
    if it % a.eval_every == 0:
        rec['val'] = round(objective(x, val=True)[1], 4)
        torch.save({'x01': x.cpu(), 'y': y.cpu(), 'it': it}, f'cache/ps_{a.tag}_it{it}.pt')
    print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
