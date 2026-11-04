"""Which weights can the images actually toggle? Replay the exact seed-42 run of a set and of K randomly perturbed copies
(+-1 level on a random share rho of the pixels, the size of what one sign-matching round does), record the net sign of
theta5 - theta0 each time. Weights whose sign changes in at least one copy are "fragile"; a round toggles them, the
others barely move (4/11: a round changes only 2-4% of net signs, and cannot pick which). Also prints the val accuracy
of each perturbed copy (noise floor of the set). Saves {name: uint8 count of sign changes} to <out>.
usage: DD_HOLDOUT=4096 PROTO_PIL=1 PROTO_EXACT=1 fragility.py <views.pt> <set.pt> <out.pt> [K=8] [rho=0.5]"""
import json, sys, torch, torch.nn.functional as F
import proto as P, dd_lib as D
V = torch.load(sys.argv[1]); x0, y = D.load_set(sys.argv[2]); out = sys.argv[3]
K = int(sys.argv[4]) if len(sys.argv) > 4 else 8; rho = float(sys.argv[5]) if len(sys.argv) > 5 else 0.5
x0, y = x0.cuda(), y.cuda()


def run(x):
    m, opt, sch = P.fresh_model(V['seed'])
    th0 = {k: p.detach().clone() for k, p in m.named_parameters()}
    for batches in V['epochs']:
        for idx, vw in batches:
            opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment_views(x[idx.cuda()], vw)), y[idx.cuda()]).backward(); opt.step()
        sch.step()
    va = P.evaluate(m, D.xva, D.yva)
    return {k: torch.sign(p.detach() - th0[k]).to(torch.int8) for k, p in m.named_parameters()}, round(va, 4)


S0, v0 = run(x0)
NT = sum(s.numel() for s in S0.values())
print(json.dumps({'base_val': v0}), flush=True)
cnt = {k: torch.zeros_like(s, dtype=torch.uint8) for k, s in S0.items()}
for s in range(K):
    g = torch.Generator(device='cuda').manual_seed(1000 + s)
    u = torch.rand(x0.shape, generator=g, device='cuda')
    d = torch.where(u < rho / 2, -1., torch.where(u < rho, 1., 0.)) / 255
    S, va = run((x0 + d).clamp(0, 1))
    ch = 0
    for k in S0:
        c = S[k] != S0[k]
        cnt[k] += c.to(torch.uint8); ch += c.sum().item()
    print(json.dumps({'copy': s, 'val': va, 'changed': round(ch / NT, 4),
                      'fragile_so_far': round(sum((c > 0).sum().item() for c in cnt.values()) / NT, 4)}), flush=True)
torch.save({k: c.cpu() for k, c in cnt.items()}, out)
