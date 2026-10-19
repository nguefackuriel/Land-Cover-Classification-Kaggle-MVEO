"""Which straight-through smoothing tau gives the most useful exact meta-gradient? For each tau: meta-gradient at the
set (exact views, fixed 2048 train images), then +-1 level on the top-k pixels by |g|; scored by the exact replay.
usage: PROTO_PIL=1 tau_probe.py <set.pt> <views.pt>"""
import json, sys, torch
src, views = sys.argv[1], sys.argv[2]
sys.argv = ['rough_exact.py', src, views]
exec(open('src/rough_exact.py').read().split('base = score(x0)')[0])
base = score(x0)
print(json.dumps({'kind': 'base', 'ce': base[0], 'train_acc': base[1], 'val': base[2]}), flush=True)
thK, saved, K = Q.forward(x0, head)
G, rl = Q.real_grad(thK, RIDX[:2048])
del thK
for tau in (0.03, 0.1, 0.3, 1.0, 3.0, 10.0):
    Q.a.tau = tau
    gx = Q.backward(x0.clone().requires_grad_(True), saved, G)
    flat = gx.flatten(); order = flat.abs().argsort(descending=True)
    for k in (10, 30, 100, 300):
        d = torch.zeros_like(flat); idx = order[:k]; d[idx] = -flat[idx].sign() / 255
        r = score((x0 + d.view_as(x0)).clamp(0, 1))
        print(json.dumps({'tau': tau, 'k': k, 'ce': r[0], 'd_ce': round(r[0] - base[0], 4), 'train_acc': r[1], 'val': r[2]}), flush=True)
