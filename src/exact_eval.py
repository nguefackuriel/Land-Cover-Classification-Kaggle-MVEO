"""Score candidate sets on the exact seed-42 run: val accuracy (4050) and accuracy on 8192 fixed training images (a second,
larger sample of the same distribution, to pick between close candidates with less noise). With DD_HOLDOUT=k the second
sample is the k held-out rows of dd_lib.HOLD instead (unseen by distillation run with the same k).
usage: PROTO_PIL=1 exact_eval.py <views.pt> <set1.pt> [<set2.pt> ...]"""
import json, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D
V = torch.load(sys.argv[1])
RIDX = torch.randperm(len(D.xtr), generator=torch.Generator().manual_seed(33))[:8192]
KEY = 'train8k'
if len(D.HOLD):                   # DD_HOLDOUT=k: score on the k held-out rows that distillation never sampled
    RIDX, KEY = D.HOLD, 'hold'
for path in sys.argv[2:]:
    x, y = D.load_set(path)
    m, opt, sch = P.fresh_model(V['seed'])
    x, y = x.cuda(), y.cuda()
    for batches in V['epochs']:
        for idx, vw in batches:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment_views(x[idx.cuda()], vw)), y[idx.cuda()]).backward(); opt.step()
        sch.step()
    va = P.evaluate(m, D.xva, D.yva); tr = P.evaluate(m, D.xtr[RIDX], D.ytr[RIDX])
    print(json.dumps({'set': path, 'val': round(va, 4), KEY: round(tr, 4), 'pooled': round((va * 4050 + tr * len(RIDX)) / (4050 + len(RIDX)), 4)}), flush=True)
