"""Accuracy of theta_0 + c * (theta_5 - theta_0) for one set (exact seed-42 run), c around 1, per parameter group.
usage: PROTO_PIL=1 PROTO_EXACT=1 DD_HOLDOUT=4096 scale_scan.py <views.pt> <set.pt>"""
import json, sys, torch, torch.nn.functional as F
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
def grp(k):
    return 'head' if k.startswith('heads') else ('embed' if k.startswith('conv_proj') or 'pos_embedding' in k or 'class_token' in k else 'body')
for name, sel in [('all', None), ('body', 'body'), ('head', 'head'), ('embed', 'embed')]:
    for c in ([0.9, 0.95, 1.05, 1.1, 1.15] if sel is None else [0.9, 1.1, 1.25]):
        m.load_state_dict({k: th0[k] + (c if sel is None or grp(k) == sel else 1.0) * d[k] for k in th0})
        va = P.evaluate(m, D.xva, D.yva); ho = P.evaluate(m, D.xtr[D.HOLD], D.ytr[D.HOLD])
        print(json.dumps({'group': name, 'c': c, 'val': round(va, 4), 'hold': round(ho, 4)}), flush=True)
