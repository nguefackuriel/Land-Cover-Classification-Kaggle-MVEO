"""Minimal-flip targets for sign matching. The near target (ceiling3) asks to flip ~15% of the signs our exact run ends
with, and the run only manages some of them; most of those flips matter little for accuracy. Here the target keeps the
sign pattern of our own run end (s_act = sign(theta5 - theta0)) and takes the near target's sign only on the top-f
fraction of the disagreeing weights, ranked by the first-order loss increase if that weight went back to our sign:
delta_i = g_i * (s_act_i - s_near_i) * R, g = grad of the train CE (fitted T, pool rows) at the near corner.
Prints the val accuracy of each corner and saves the targets as state dicts (theta0 + R * s on the parameters) that
signmatch.py --target reads like a ceiling3 file.
usage: DD_HOLDOUT=4096 PROTO_PIL=1 flip_target.py <near.pt> <run end state.pt> <out prefix> [fractions] [near|act]
2/11 result (224, KD2 + near_DG1): ranked at the near corner, partial flips HURT (our corner 87.5 -> 80-84 for f 0.05-0.5,
92.5 only at f=1): the near target's flips only work together."""
import json, math, sys, torch, torch.nn.functional as F
from torch.func import functional_call
import proto as P, dd_lib as D

near_p, act_p, out = sys.argv[1:4]
fracs = [float(f) for f in (sys.argv[4] if len(sys.argv) > 4 else '0.05,0.1,0.25,0.5,1').split(',')]
AT = sys.argv[5] if len(sys.argv) > 5 else 'near'   # 'act': rank flips by the gradient at our own corner instead
dev = 'cuda'
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]; WD = 0.05; R = sum(LRS)
m, _, _ = P.fresh_model(42); m.eval()
for p in m.parameters():
    p.requires_grad_(False)
th0 = {k: p.detach().clone() for k, p in m.named_parameters()}
near = torch.load(near_p, map_location=dev); act = torch.load(act_p, map_location=dev)
act = act.get('model', act) if isinstance(act, dict) and 'model' in act else act
s_near = {k: torch.sign(near[k].float() - th0[k]) for k in th0}
s_act = {k: torch.sign(act[k].float() - th0[k]) for k in th0}
del near, act


def corner(S):
    """End of the official path when every step moves by -lr*(-s) exactly (as in signmatch.py)."""
    th = dict(th0)
    for lr in LRS:
        th = {k: t * (1 - lr * WD) + lr * S[k] for k, t in th.items()}
    return th


@torch.no_grad()
def val_acc(S):
    th = corner(S)
    sd = m.state_dict()
    saved = {k: sd[k].clone() for k in th}
    for k, t in th.items():
        sd[k].copy_(t)
    acc = P.evaluate(m, D.xva, D.yva)
    for k, t in saved.items():
        sd[k].copy_(t)
    return acc


# gradient of the fitted-temperature train CE at the near corner (or ours, AT=act), pool rows only
cn = corner(s_near if AT == 'near' else s_act)
det = {k: t.detach().requires_grad_(True) for k, t in cn.items()}
ridx = D.sample_train(2048, torch.Generator().manual_seed(9))
yb = D.ytr[ridx].to(dev)
with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
    o = torch.cat([functional_call(m, cn, (P.to_eval(D.xtr[ridx[j:j + 256]].to(dev)),)).float() for j in range(0, 2048, 256)])
Ts = torch.logspace(-2, 1, 61, device=dev)
T = Ts[torch.stack([F.cross_entropy(o / t, yb) for t in Ts]).argmin()].item()
G = {k: torch.zeros_like(t) for k, t in det.items()}
for j in range(0, 2048, 128):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        l = F.cross_entropy(functional_call(m, det, (P.to_eval(D.xtr[ridx[j:j + 128]].to(dev)),)).float() / T, yb[j:j + 128])
    for (k, _), g in zip(det.items(), torch.autograd.grad(l, list(det.values()))):
        G[k] += g.float()
del det, cn

dis = {k: (s_near[k] != s_act[k]) & (s_near[k] != 0) for k in th0}
# delta = how much worse the corner gets without this flip (near: going back at the near corner; act: -gain of the flip
# at our corner), so larger = more useful flip in both cases
delta = {k: torch.where(dis[k], G[k] * (s_act[k] - s_near[k]) * R, torch.full_like(G[k], -float('inf'))) for k in th0}
n_dis = sum(d.sum().item() for d in dis.values()); NT = sum(t.numel() for t in th0.values())
allv = torch.cat([d[dis[k]].flatten() for k, d in delta.items()])
print(json.dumps({'disagree_frac': round(n_dis / NT, 4), 'T': T, 'pos_delta_frac': round((allv > 0).float().mean().item(), 4),
                  'acc_act_corner': round(val_acc(s_act), 4), 'acc_near_corner': round(val_acc(s_near), 4)}), flush=True)
srt = allv.sort(descending=True).values
del allv
for f in fracs:
    kth = srt[min(len(srt) - 1, max(0, int(f * len(srt)) - 1))].item()
    S = {k: torch.where(dis[k] & (delta[k] >= kth), s_near[k], s_act[k]) for k in th0}
    acc = val_acc(S)
    print(json.dumps({'frac': f, 'flips': round(f * n_dis / NT, 4), 'acc_corner': round(acc, 4)}), flush=True)
    th = corner(S)
    sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
    for k, t in th.items():
        sd[k] = t.cpu()
    torch.save(sd, f'{out}_f{f:g}.pt')
