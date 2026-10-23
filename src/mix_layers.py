"""usage: PROTO_PIL=1 python mix_layers.py <near target .pt> <end state of the exact run .pt>"""
# Which layers' sign disagreements cost the most: put the corner values into the actual end state, one group at a time.
import sys, math, torch
import dd_lib as D, proto as P
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]; WD = 0.05
m, _, _ = P.fresh_model(42); m.eval()
th0 = {k: p.detach().clone() for k, p in m.named_parameters()}
t = torch.load(sys.argv[1]); act = torch.load(sys.argv[2])
act = {k: (act[k] if k in act else act['model'][k]).cuda().float() for k in th0} if not all(k in act for k in th0) else {k: act[k].cuda().float() for k in th0}
S = {k: torch.sign(t[k].cuda() - th0[k]) for k in th0}
cor = dict(th0)
for lr in LRS:
    cor = {k: v * (1 - lr * WD) + lr * S[k] for k, v in cor.items()}
def ev(th):
    with torch.no_grad():
        for k, p in m.named_parameters(): p.copy_(th[k])
    return round(P.evaluate(m, D.xva, D.yva), 4)
agree = lambda ks: sum((torch.sign(act[k] - th0[k]) == S[k]).sum().item() for k in ks) / sum(th0[k].numel() for k in ks)
def group(k):
    if k.startswith('encoder.layers.encoder_layer_'):
        i = int(k.split('_')[2].split('.')[0]); return 'blocks%02d-%02d' % (i // 3 * 3, i // 3 * 3 + 2)
    if k.startswith('heads'): return 'head'
    if k.startswith('encoder.ln'): return 'final_ln'
    return 'embed'
G = {}
for k in th0: G.setdefault(group(k), []).append(k)
print('actual', ev(act), 'agree', round(agree(list(th0)), 3), flush=True)
print('corner', ev(cor), flush=True)
for g, ks in sorted(G.items()):
    mix = dict(act); mix.update({k: cor[k] for k in ks})
    sub = [k for k in ks if ('attn' in k)]; 
    print(g, 'n=%.1fM' % (sum(th0[k].numel() for k in ks) / 1e6), 'agree', round(agree(ks), 3), 'acc with corner', ev(mix), flush=True)
for part in ['self_attention', 'mlp', 'ln_']:
    ks = [k for k in th0 if part in k]
    mix = dict(act); mix.update({k: cor[k] for k in ks})
    print('all', part, 'n=%.1fM' % (sum(th0[k].numel() for k in ks) / 1e6), 'agree', round(agree(ks), 3), 'acc with corner', ev(mix), flush=True)
