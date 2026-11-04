"""Corner climbing: a target close to our own run that is still much better. Start from the sign pattern of our run end
(s_act) and flip weights in small chunks, re-linearising after each chunk: gradient of the train CE (fitted T, pool rows)
at the current corner, flip the top chunk by first-order gain, repeat. corner_mix.py / bias_diag.py (4/11): one big
ranked chunk collapses (class priors shift: 224 5% ranked flips 79.95, 90.17 after per-class offsets), random flips are
smooth; small re-linearised chunks should keep the gain without the shift.
Candidates: weights where the near target disagrees with s_act (--near) or all weights (--free); optional --undecided q:
only weights whose run displacement |theta5 - theta0| / R is below q (the run barely commits to their sign).
Prints val / hold of each corner; saves the target (corner state dict, a --target file) at the asked iterations.
usage: DD_HOLDOUT=4096 PROTO_PIL=1 climb_target.py <run end state.pt> <out prefix> --near <near.pt> --chunk 0.001
       --iters 20 --save 5,10,20"""
import argparse, json, math, sys, torch, torch.nn.functional as F
from torch.func import functional_call
import proto as P, dd_lib as D

ap = argparse.ArgumentParser()
ap.add_argument('act'); ap.add_argument('out')
ap.add_argument('--near', default='')
ap.add_argument('--chunk', type=float, default=0.001)    # share of all weights flipped per iteration
ap.add_argument('--iters', type=int, default=20)
ap.add_argument('--save', default='')
ap.add_argument('--undecided', type=float, default=0.)   # > 0: only weights with |theta5 - theta0| / R below this
ap.add_argument('--bs', type=int, default=2048)
ap.add_argument('--mask', default='')       # fragility.py counts: only weights whose sign changed in >= --mask-min copies
ap.add_argument('--mask-min', type=int, default=1)
a = ap.parse_args()
save = {int(s) for s in a.save.split(',') if s}
dev = 'cuda'
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]; WD = 0.05; R = sum(LRS)
m, _, _ = P.fresh_model(42); m.eval()
for p in m.parameters():
    p.requires_grad_(False)
th0 = {k: p.detach().clone() for k, p in m.named_parameters()}
act = torch.load(a.act, map_location=dev)
act = act.get('model', act) if isinstance(act, dict) and 'model' in act else act
S = {k: torch.sign(act[k].float() - th0[k]) for k in th0}
S = {k: torch.where(s == 0, torch.ones_like(s), s) for k, s in S.items()}
cand = {k: torch.ones_like(s, dtype=torch.bool) for k, s in S.items()}
if a.undecided > 0:
    cand = {k: c & ((act[k].float() - th0[k]).abs() / R < a.undecided) for k, c in cand.items()}
del act
if a.mask:
    _c = torch.load(a.mask, map_location=dev)
    cand = {k: c & (_c[k] >= a.mask_min) for k, c in cand.items()}
    del _c
if a.near:
    near = torch.load(a.near, map_location=dev)
    cand = {k: c & (torch.sign(near[k].float() - th0[k]) == -S[k]) for k, c in cand.items()}
    del near
NT = sum(t.numel() for t in th0.values())
print(json.dumps({'candidates': round(sum(c.sum().item() for c in cand.values()) / NT, 4)}), flush=True)


def corner(S):
    th = {}
    for k, t in th0.items():
        for lr in LRS:
            t = t * (1 - lr * WD) + lr * S[k]
        th[k] = t
    return th


@torch.no_grad()
def accs(th):
    sd = m.state_dict(); saved = {k: sd[k].clone() for k in th0}
    for k, t in th.items():
        sd[k].copy_(t)
    va = P.evaluate(m, D.xva, D.yva); ho = P.evaluate(m, D.xtr[D.HOLD], D.ytr[D.HOLD]) if len(D.HOLD) else -1
    for k, t in saved.items():
        sd[k].copy_(t)
    return round(va, 4), round(ho, 4)


flipped = 0
for it in range(a.iters + 1):
    th = corner(S)
    va, ho = accs(th)
    print(json.dumps({'it': it, 'flipped': round(flipped / NT, 5), 'val': va, 'hold': ho}), flush=True)
    if it in save:
        sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        for k, t in th.items():
            sd[k] = t.cpu()
        torch.save(sd, f'{a.out}_it{it}.pt')
    if it == a.iters:
        break
    ridx = D.sample_train(a.bs, torch.Generator().manual_seed(100 + it))
    yb = D.ytr[ridx].to(dev)
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = torch.cat([functional_call(m, th, (P.to_eval(D.xtr[ridx[j:j + 256]].to(dev)),)).float() for j in range(0, a.bs, 256)])
    Ts = torch.logspace(-2, 1, 61, device=dev)
    T = Ts[torch.stack([F.cross_entropy(o / t, yb) for t in Ts]).argmin()].item()
    det = {k: t.detach().requires_grad_(True) for k, t in th.items()}
    G = {k: torch.zeros_like(t) for k, t in det.items()}
    for j in range(0, a.bs, 128):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = F.cross_entropy(functional_call(m, det, (P.to_eval(D.xtr[ridx[j:j + 128]].to(dev)),)).float() / T, yb[j:j + 128])
        for (k, _), g in zip(det.items(), torch.autograd.grad(l, list(det.values()))):
            G[k] += g.float()
    del det, th
    gain = {k: torch.where(cand[k], G[k] * S[k], torch.full_like(G[k], -float('inf'))) for k in th0}   # > 0: flip helps
    k_n = int(a.chunk * NT)
    allg = torch.cat([g.flatten() for g in gain.values()])
    kth = allg.topk(k_n).values[-1].item()
    del allg
    kth = max(kth, 0.)
    n_it = 0
    for k in th0:
        f = gain[k] > kth
        S[k] = torch.where(f, -S[k], S[k]); cand[k] = cand[k] & ~f
        n_it += f.sum().item()
    flipped += n_it
    del G, gain
