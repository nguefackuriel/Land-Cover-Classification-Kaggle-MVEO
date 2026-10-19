"""Replay the official seed-42 run of a set with the exact PIL training inputs (proto.augment_views_exact) instead of
the emulation, and print the per-epoch validation accuracy (fp32 eval). usage: PROTO_PIL=1 exact_pil_check.py <set> <views>"""
import json, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D


@torch.no_grad()
def f32_eval(m, xu8, y, bs=256):
    m.eval(); correct = 0
    for i in range(0, len(xu8), bs):
        correct += (m(P.to_eval(xu8[i:i + bs].cuda())).argmax(1).cpu() == y[i:i + bs]).sum().item()
    m.train(); return correct / len(xu8)


x, y = D.load_set(sys.argv[1]); V = torch.load(sys.argv[2])
for mode in ('emulated', 'exact_pil'):
    m, opt, sch = P.fresh_model(V['seed']); xs, ys = x.cuda(), y.cuda(); accs = []
    for batches in V['epochs']:
        for idx, vw in batches:
            xb = P.augment_views(xs[idx.cuda()], vw) if mode == 'emulated' else P.augment_views_exact(xs[idx.cuda()], vw)
            opt.zero_grad(set_to_none=True); F.cross_entropy(m(xb), ys[idx.cuda()]).backward(); opt.step()
        sch.step(); accs.append(round(f32_eval(m, D.xva, D.yva), 4))
    print(json.dumps({'set': sys.argv[1], 'mode': mode, 'per_epoch': accs}), flush=True)
