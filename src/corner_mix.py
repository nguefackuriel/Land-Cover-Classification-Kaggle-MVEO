"""Is the space between our corner and the near-target corner rugged, or only for ranked flips? flip_target.py showed that
flipping the top-ranked disagreements hurts (224: 88.84 -> 80 at 5%). Here the flipped subset is random (several
fractions, two seeds), then whole blocks (near signs inside one block only, and near signs everywhere except one block).
Prints val accuracy of each corner and the disagreement rate per block.
usage: DD_HOLDOUT=4096 PROTO_PIL=1 corner_mix.py <near.pt> <run end state.pt> [fractions]"""
import json, math, re, sys, torch
import proto as P, dd_lib as D

near_p, act_p = sys.argv[1:3]
fracs = [float(f) for f in (sys.argv[3] if len(sys.argv) > 3 else '0.1,0.25,0.5,0.75,0.9').split(',')]
dev = 'cuda'
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]; WD = 0.05
m, _, _ = P.fresh_model(42); m.eval()
for p in m.parameters():
    p.requires_grad_(False)
th0 = {k: p.detach().clone() for k, p in m.named_parameters()}
near = torch.load(near_p, map_location=dev); act = torch.load(act_p, map_location=dev)
act = act.get('model', act) if isinstance(act, dict) and 'model' in act else act
s_near = {k: torch.sign(near[k].float() - th0[k]) for k in th0}
s_act = {k: torch.sign(act[k].float() - th0[k]) for k in th0}
del near, act


def group(k):
    g = re.search(r'encoder_layer_(\d+)', k)
    if g:
        return 'L%02d' % int(g.group(1))
    return 'embed' if k.startswith('conv_proj') or 'pos_embedding' in k or 'class_token' in k else 'top'


@torch.no_grad()
def val_acc(S):
    sd = m.state_dict()
    saved = {k: sd[k].clone() for k in th0}
    for k, t in th0.items():
        th = t
        for lr in LRS:
            th = th * (1 - lr * WD) + lr * S[k]
        sd[k].copy_(th)
    acc = P.evaluate(m, D.xva, D.yva)
    for k, t in saved.items():
        sd[k].copy_(t)
    return round(acc, 4)


def out(**kw):
    print(json.dumps(kw), flush=True)


dis = {k: (s_near[k] != s_act[k]) & (s_near[k] != 0) for k in th0}
NT = sum(t.numel() for t in th0.values())
out(disagree=round(sum(d.sum().item() for d in dis.values()) / NT, 4), act=val_acc(s_act), near=val_acc(s_near))
groups = sorted(set(group(k) for k in th0))
for gname in groups:
    ks = [k for k in th0 if group(k) == gname]
    n = sum(th0[k].numel() for k in ks); d = sum(dis[k].sum().item() for k in ks)
    only = {k: (s_near[k] if group(k) == gname else s_act[k]) for k in th0}
    but = {k: (s_act[k] if group(k) == gname else s_near[k]) for k in th0}
    out(block=gname, params=n, disagree=round(d / n, 4), near_only_here=val_acc(only), near_except_here=val_acc(but))
for seed in (0, 1):
    gen = torch.Generator(device=dev).manual_seed(seed)
    U = {k: torch.rand(t.shape, generator=gen, device=dev) for k, t in th0.items()}
    for f in fracs:
        S = {k: torch.where(dis[k] & (U[k] < f), s_near[k], s_act[k]) for k in th0}
        out(seed=seed, frac=f, acc=val_acc(S))
