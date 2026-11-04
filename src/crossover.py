"""Per-image crossover between sibling sets of one lineage (same views, same row order): replace the image of one row of
the base set by the same row of a donor set, replay the exact seed-42 run, score val + hold. Pixel soups (averages) did
nothing; whole-image swaps keep each image intact.
usage: DD_HOLDOUT=4096 PROTO_PIL=1 crossover.py <views.pt> <base.pt> <donor1.pt> [<donor2.pt> ...]"""
import json, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D
V = torch.load(sys.argv[1])
RIDX = D.HOLD if len(D.HOLD) else torch.randperm(len(D.xtr), generator=torch.Generator().manual_seed(33))[:8192]


def score(x, y):
    m, opt, sch = P.fresh_model(V['seed'])
    for batches in V['epochs']:
        for idx, vw in batches:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment_views(x[idx.cuda()], vw)), y[idx.cuda()]).backward(); opt.step()
        sch.step()
    va = P.evaluate(m, D.xva, D.yva); tr = P.evaluate(m, D.xtr[RIDX], D.ytr[RIDX])
    return round(va, 4), round(tr, 4), round((va * len(D.yva) + tr * len(RIDX)) / (len(D.yva) + len(RIDX)), 4)


xb, yb = D.load_set(sys.argv[2]); xb, yb = xb.cuda(), yb.cuda()
v, h, p = score(xb, yb)
print(json.dumps({'base': sys.argv[2], 'val': v, 'hold': h, 'pooled': p}), flush=True)
for dp in sys.argv[3:]:
    xd, yd = D.load_set(dp); xd, yd = xd.cuda(), yd.cuda()
    assert torch.equal(yd, yb), 'donor rows do not match the base labels'
    for r in range(len(yb)):
        if torch.equal(xd[r], xb[r]):
            continue
        x = xb.clone(); x[r] = xd[r]
        v, h, p = score(x, yb)
        print(json.dumps({'donor': dp, 'row': r, 'label': int(yb[r]), 'val': v, 'hold': h, 'pooled': p}), flush=True)
