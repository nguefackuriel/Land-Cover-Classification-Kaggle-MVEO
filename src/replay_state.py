"""Replay the exact official seed-42 run of a set (recorded views) and save the final state dict + val accuracy.
usage: PROTO_PIL=1 replay_state.py <set.pt> <views.pt> <out.pt>"""
import sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D
x, y = D.load_set(sys.argv[1]); V = torch.load(sys.argv[2])
m, opt, sch = P.fresh_model(V['seed'])
x, y = x.cuda(), y.cuda()
for batches in V['epochs']:
    for idx, vw in batches:
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(P.augment_views(x[idx.cuda()], vw)), y[idx.cuda()]).backward(); opt.step()
    sch.step()
print('val', P.evaluate(m, D.xva, D.yva))
torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, sys.argv[3])
