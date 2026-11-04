"""Corner path from our run to the near target by whole blocks. corner_mix.py (4/11) showed that taking the near signs
inside one block helps (224: L09 alone 88.84 -> 89.70), unlike ranked single-weight flips. Here blocks are added one by
one in the given order (cumulative), the val accuracy of each corner is printed, and the hybrid target (theta0 + path
with near signs on the chosen blocks, our signs elsewhere) is saved for the k values asked, as a --target file.
usage: DD_HOLDOUT=4096 PROTO_PIL=1 block_path.py <near.pt> <run end state.pt> <order: L09,L10,...> <out prefix> [save ks]"""
import json, math, re, sys, torch
import proto as P, dd_lib as D

near_p, act_p, order, out = sys.argv[1:5]
order = order.split(',')
save = [int(k) for k in sys.argv[5].split(',')] if len(sys.argv) > 5 else []
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


def corner(S):
    th = {}
    for k, t in th0.items():
        for lr in LRS:
            t = t * (1 - lr * WD) + lr * S[k]
        th[k] = t
    return th


@torch.no_grad()
def val_acc(th):
    sd = m.state_dict()
    saved = {k: sd[k].clone() for k in th0}
    for k, t in th.items():
        sd[k].copy_(t)
    acc = P.evaluate(m, D.xva, D.yva)
    for k, t in saved.items():
        sd[k].copy_(t)
    return round(acc, 4)


for n in range(len(order) + 1):
    B = set(order[:n])
    S = {k: (s_near[k] if group(k) in B else s_act[k]) for k in th0}
    th = corner(S)
    print(json.dumps({'k': n, 'blocks': order[:n], 'acc': val_acc(th)}), flush=True)
    if n in save:
        sd = {k: v.detach().cpu() for k, v in m.state_dict().items()}
        for k, t in th.items():
            sd[k] = t.cpu()
        torch.save(sd, f'{out}_k{n}.pt')
