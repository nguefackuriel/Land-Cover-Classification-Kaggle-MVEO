"""Is 'move 5% further' reachable? Ratio |d|/R per weight for theta_5 - theta_0 (R = sum of the official lr's, the AdamW
box), then accuracy of theta_0 + clip(c*d, -R, R) and the real CE at the fitted temperature along c.
usage: PROTO_PIL=1 PROTO_EXACT=1 DD_HOLDOUT=4096 box_scan.py <views.pt> <set.pt>"""
import json, math, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D
V = torch.load(sys.argv[1])
x, y = D.load_set(sys.argv[2])
m, opt, sch = P.fresh_model(V['seed'])
th0 = {k: t.detach().clone() for k, t in m.state_dict().items()}
x, y = x.cuda(), y.cuda()
for batches in V['epochs']:
    for idx, vw in batches:
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(P.augment_views(x[idx.cuda()], vw)), y[idx.cuda()]).backward(); opt.step()
    sch.step()
d = {k: t.detach().clone() - th0[k] for k, t in m.state_dict().items()}
R = sum(3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5))
r = torch.cat([v.abs().flatten() / R for v in d.values()])
qs = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9, 0.99], device=r.device)
print(json.dumps({'R': R, 'mean_ratio': round(r.mean().item(), 4), 'quantiles': [round(v, 4) for v in torch.quantile(r[torch.randperm(len(r), device=r.device)[:2_000_000]], qs).tolist()],
                  'frac_ge_0.95': round((r >= 0.95).float().mean().item(), 4), 'frac_ge_0.99': round((r >= 0.99).float().mean().item(), 4)}), flush=True)
del r
ridx = D.HOLD
def real_ce(model):
    with torch.no_grad():
        o = torch.cat([model(P.to_eval(D.xtr[ridx[j:j + 256]].cuda())).float() for j in range(0, len(ridx), 256)])
    yb = D.ytr[ridx].cuda()
    Ts = torch.logspace(-2, 1, 61, device='cuda')
    ces = torch.stack([F.cross_entropy(o / t, yb) for t in Ts])
    return round(ces.min().item(), 4), round(Ts[ces.argmin()].item(), 4), round(F.cross_entropy(o, yb).item(), 4)
for c in (1.0, 1.02, 1.05, 1.08):
    for clip in (False, True):
        if c == 1.0 and clip:
            continue
        m.load_state_dict({k: th0[k] + ((c * d[k]).clamp(-R, R) if clip else c * d[k]) for k in th0})
        va = P.evaluate(m, D.xva, D.yva); ho = P.evaluate(m, D.xtr[D.HOLD], D.ytr[D.HOLD])
        ce, T, ce1 = real_ce(m)
        print(json.dumps({'c': c, 'clip_box': clip, 'val': round(va, 4), 'hold': round(ho, 4), 'ce_fitT': ce, 'T': T, 'ce_T1': ce1}), flush=True)
