"""Score candidate sets on the exact seed-42 run: val accuracy (4050) and accuracy on 8192 fixed training images (a second,
larger sample of the same distribution, to pick between close candidates with less noise).
usage: PROTO_PIL=1 exact_eval.py <views.pt> <set1.pt> [<set2.pt> ...]"""
import json, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D
V = torch.load(sys.argv[1])
RIDX = torch.randperm(len(D.xtr), generator=torch.Generator().manual_seed(33))[:8192]
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
    print(json.dumps({'set': path, 'val': round(va, 4), 'train8k': round(tr, 4), 'pooled': round((va * 4050 + tr * 8192) / 12242, 4)}), flush=True)
