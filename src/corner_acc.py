"""usage: PROTO_PIL=1 python corner_acc.py <near target .pt>"""
# val accuracy of a near target, of its single-sign corner, and of a partial corner (sign only where the target is at the box edge)
import sys, math, torch
import dd_lib as D, proto as P
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]; WD = 0.05; R = sum(LRS)
m, _, _ = P.fresh_model(42); m.eval()
th0 = {k: p.detach().clone() for k, p in m.named_parameters()}
t = torch.load(sys.argv[1])
def ev(th):
    with torch.no_grad():
        for k, p in m.named_parameters(): p.copy_(th[k])
    return P.evaluate(m, D.xva, D.yva)
print('target', round(ev({k: t[k].cuda() for k in th0}), 4), flush=True)
S = {k: torch.sign(t[k].cuda() - th0[k]) for k in th0}
th = dict(th0)
for lr in LRS:
    th = {k: v * (1 - lr * WD) + lr * S[k] for k, v in th.items()}
print('corner', round(ev(th), 4), flush=True)
edge = sum(((t[k].cuda() - th0[k]).abs() > 0.99 * R).sum().item() for k in th0) / sum(v.numel() for v in th0.values())
print('edge share', round(edge, 3))
