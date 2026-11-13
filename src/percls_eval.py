"""Per-class recall and confusion counts of the exact seed-42 run, pooled over val (4050) and the held-out block.
Same training as exact_eval.py / crossover.py. Used to see which classes make one start set better than another.
usage: PROTO_PIL=1 PROTO_EXACT=1 DD_HOLDOUT=4096 percls_eval.py <views.pt> <set1.pt> [<set2.pt> ...]"""
import json, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D
V = torch.load(sys.argv[1])
assert len(D.HOLD), 'set DD_HOLDOUT=4096'


def train(x, y):
    m, opt, sch = P.fresh_model(V['seed'])
    for batches in V['epochs']:
        for idx, vw in batches:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment_views(x[idx.cuda()], vw)), y[idx.cuda()]).backward(); opt.step()
        sch.step()
    return m


@torch.no_grad()
def preds(m, bs=256):
    m.eval(); out = []
    xv = P._exact_val() if P.EXACT else None
    if xv is not None:
        for i in range(0, len(xv), bs):
            out.append(m(xv[i:i + bs].cuda(non_blocking=True)).argmax(1).cpu())
    else:
        for i in range(0, len(D.xva), bs):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                out.append(m(P.to_eval(D.xva[i:i + bs].cuda())).argmax(1).cpu())
    xh = D.xtr[D.HOLD]
    for i in range(0, len(xh), bs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            out.append(m(P.to_eval(xh[i:i + bs].cuda())).argmax(1).cpu())
    m.train()
    return torch.cat(out)


ytrue = torch.cat([D.yva, D.ytr[D.HOLD]])
for path in sys.argv[2:]:
    x, y = D.load_set(path)
    m = train(x.cuda(), y.cuda())
    p = preds(m)
    conf = torch.zeros(10, 10, dtype=torch.long)
    conf.index_put_((ytrue, p), torch.ones_like(p), accumulate=True)
    rec = (conf.diag().float() / conf.sum(1).float()).tolist()
    nv = len(D.yva)
    print(json.dumps({'set': path, 'val': round((p[:nv] == ytrue[:nv]).float().mean().item(), 4),
                      'hold': round((p[nv:] == ytrue[nv:]).float().mean().item(), 4),
                      'pooled': round((p == ytrue).float().mean().item(), 4),
                      'recall': [round(r, 4) for r in rec], 'conf': conf.tolist()}), flush=True)
    del m; torch.cuda.empty_cache()
