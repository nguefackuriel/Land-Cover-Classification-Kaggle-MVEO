"""Sparse uint8 moves along the exact meta-gradient: +-1 level on the top-k pixels by |g| (and controls: opposite sign,
random k pixels), scored by the exact replay (calibrated CE and acc on 4096 fixed train images, val acc).
usage: PROTO_PIL=1 sparse_exact.py <set.pt> <views.pt>"""
import json, sys, torch
src, views = sys.argv[1], sys.argv[2]
sys.argv = ['rough_exact.py', src, views]
exec(open('src/rough_exact.py').read().split('base = score(x0)')[0])
base = score(x0)
print(json.dumps({'kind': 'base', 'ce': base[0], 'train_acc': base[1], 'val': base[2]}), flush=True)
thK, saved, K = Q.forward(x0, head)
G, rl = Q.real_grad(thK, RIDX[:2048])
del thK
gx = Q.backward(x0.clone().requires_grad_(True), saved, G)
del saved, G
flat = gx.flatten(); order = flat.abs().argsort(descending=True)
gen = torch.Generator(device=dev).manual_seed(5)
for k in (3, 10, 30, 100, 300, 1000, 3000):
    for kind in ('top', 'top_opp', 'rand'):
        if kind == 'rand' and k not in (30, 300):
            continue
        idx = order[:k] if kind != 'rand' else torch.randperm(flat.numel(), generator=gen, device=dev)[:k]
        d = torch.zeros_like(flat)
        sgn = -flat[idx].sign() if kind != 'rand' else (torch.randint(0, 2, (k,), generator=gen, device=dev).float() * 2 - 1)
        if kind == 'top_opp':
            sgn = -sgn
        d[idx] = sgn / 255
        xk = (x0 + d.view_as(x0)).clamp(0, 1)
        r = score(xk)
        print(json.dumps({'kind': kind, 'k': k, 'ce': r[0], 'train_acc': r[1], 'val': r[2]}), flush=True)
