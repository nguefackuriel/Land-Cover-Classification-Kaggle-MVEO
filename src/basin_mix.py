"""Combine two distilled sets class by class: per-class validation accuracy of each set (replica, 6 seeds), then a mix
that takes each class's images from the set where that class scores higher, evaluated on the same 6 seeds.
Both sets must hold the same number of images per class. Saves cache/mix_<tag>.pt.
usage: basin_mix.py cache/A.pt cache/B.pt tag"""
import json, sys, torch
import torch.nn.functional as F
import dd_lib as D, proto as P

SEEDS = (42, 0, 1, 2, 3, 4)


def train_preds(x01, y, seed):
    m, opt, sch = P.fresh_model(seed); x01, y = x01.cuda(), y.cuda()
    for ep in range(5):
        perm = torch.randperm(len(y))
        for j in range(0, len(y), 32):
            b = perm[j:j + 32]; opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment(x01[b])), y[b]).backward(); opt.step()
        sch.step()
    m.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(D.xva), 256):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                preds.append(m(P.to_eval(D.xva[i:i + 256].cuda())).argmax(1).cpu())
    return torch.cat(preds)


def per_class(x, y):
    accs, overall = torch.zeros(10), []
    for s in SEEDS:
        p = train_preds(x, y, s)
        accs += torch.stack([(p[D.yva == c] == c).float().mean() for c in range(10)]) / len(SEEDS)
        overall.append((p == D.yva).float().mean().item())
    return accs, overall


A, B = [dict(zip(('x01', 'y'), D.load_set(f))) for f in sys.argv[1:3]]
tag = sys.argv[3]
assert torch.equal(torch.bincount(A['y'], minlength=10), torch.bincount(B['y'], minlength=10))
ca, oa = per_class(A['x01'], A['y'])
cb, ob = per_class(B['x01'], B['y'])
pick = (cb > ca).tolist()
xs, ys = [], []
for c in range(10):
    S = B if pick[c] else A
    xs.append(S['x01'][S['y'] == c]); ys.append(S['y'][S['y'] == c])
xm, ym = torch.cat(xs), torch.cat(ys)
cm, om = per_class(xm, ym)
torch.save({'x01': xm, 'y': ym, 'pick_B': pick}, f'cache/mix_{tag}.pt')
r = lambda v: [round(float(t) * 100, 2) for t in v]
print(json.dumps({'A': sys.argv[1], 'B': sys.argv[2], 'per_class_A': r(ca), 'per_class_B': r(cb), 'pick_B': pick,
                  'per_class_mix': r(cm), 'overall_A': r(oa), 'overall_B': r(ob), 'overall_mix': r(om),
                  'mean_A': round(sum(oa) / 6 * 100, 2), 'mean_B': round(sum(ob) / 6 * 100, 2),
                  'mean_mix': round(sum(om) / 6 * 100, 2)}), flush=True)
