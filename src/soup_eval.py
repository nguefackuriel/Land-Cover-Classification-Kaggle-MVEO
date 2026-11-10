"""Weight-space soup of exact seed-42 runs: does averaging the end states of several good sets beat each one?
For each set: exact official run (as exact_eval.py) -> displacement d_i = theta_5 - theta_0 (kept on CPU).
Then evaluate mixes: mean of all, mean of a subset, scaled means, majority sign x mean |d|.
usage: PROTO_PIL=1 PROTO_EXACT=1 DD_HOLDOUT=4096 soup_eval.py <views.pt> <set1.pt> <set2.pt> ..."""
import json, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D

V = torch.load(sys.argv[1])
paths = sys.argv[2:]
RIDX, KEY = (D.HOLD, 'hold') if len(D.HOLD) else (torch.randperm(len(D.xtr), generator=torch.Generator().manual_seed(33))[:8192], 'train8k')


def run(path):
    x, y = D.load_set(path)
    m, opt, sch = P.fresh_model(V['seed'])
    x, y = x.cuda(), y.cuda()
    for batches in V['epochs']:
        for idx, vw in batches:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment_views(x[idx.cuda()], vw)), y[idx.cuda()]).backward(); opt.step()
        sch.step()
    return m


def score(m, tag):
    va = P.evaluate(m, D.xva, D.yva); tr = P.evaluate(m, D.xtr[RIDX], D.ytr[RIDX])
    rec = {'mix': tag, 'val': round(va, 4), KEY: round(tr, 4), 'pooled': round((va * 4050 + tr * len(RIDX)) / (4050 + len(RIDX)), 4)}
    print(json.dumps(rec), flush=True)
    return rec


m0, _, _ = P.fresh_model(V['seed'])
th0 = {k: p.detach().cpu().clone() for k, p in m0.state_dict().items()}
del m0
deltas = []
for p in paths:
    m = run(p)
    score(m, p.split('/')[-1])
    deltas.append({k: (t.detach().cpu() - th0[k]) for k, t in m.state_dict().items()})
    del m
    torch.cuda.empty_cache()

mm, _, _ = P.fresh_model(V['seed'])


def load_mix(d, tag):
    mm.load_state_dict({k: th0[k] + d[k] for k in th0})
    return score(mm, tag)


n = len(deltas)
mean = {k: sum(d[k] for d in deltas) / n for k in th0}
for c in (1.0, 1.25, 1.5):
    load_mix({k: c * t for k, t in mean.items()}, f'mean_all x{c}')
# majority sign with the mean magnitude of the members
maj = {k: torch.sign(sum(torch.sign(d[k]) for d in deltas)) * (sum(d[k].abs() for d in deltas) / n) for k in th0}
load_mix(maj, 'majority_sign x mean|d|')
# pairs with the first set (the best one)
for j in range(1, n):
    load_mix({k: (deltas[0][k] + deltas[j][k]) / 2 for k in th0}, f'pair 0+{j}')
# sign agreement between members
for j in range(1, n):
    agree = sum((torch.sign(deltas[0][k]) == torch.sign(deltas[j][k])).sum().item() for k in th0)
    tot = sum(t.numel() for t in th0.values())
    print(json.dumps({'agree_0_vs': j, 'frac': round(agree / tot, 4)}), flush=True)
