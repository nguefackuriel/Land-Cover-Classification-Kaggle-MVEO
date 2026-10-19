"""Exact replay of a set (emulated inputs), then fp32 validation accuracy with eval batch sizes 32 (as baseline.py)
and 256 (our replica), at every epoch. usage: PROTO_PIL=1 evalbs_check.py <set> <views>"""
import json, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D


@torch.no_grad()
def f32_eval(m, xu8, y, bs):
    m.eval(); correct = 0
    for i in range(0, len(xu8), bs):
        correct += (m(P.to_eval(xu8[i:i + bs].cuda())).argmax(1).cpu() == y[i:i + bs]).sum().item()
    m.train(); return round(correct / len(xu8), 4)


x, y = D.load_set(sys.argv[1]); V = torch.load(sys.argv[2])
m, opt, sch = P.fresh_model(V['seed']); xs, ys = x.cuda(), y.cuda(); out = []
for batches in V['epochs']:
    for idx, vw in batches:
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(P.augment_views(xs[idx.cuda()], vw)), ys[idx.cuda()]).backward(); opt.step()
    sch.step(); out.append((f32_eval(m, D.xva, D.yva, 32), f32_eval(m, D.xva, D.yva, 256)))
print(json.dumps({'set': sys.argv[1], 'per_epoch_bs32_bs256': out}))
