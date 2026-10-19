"""Best network reachable by the official recipe in K steps (seed-42 head): K Adam steps move each weight by at most
about sum(lr), so we search that L-inf box around the ImageNet weights with projected sign descent on real training
images (augmented as in training) and report validation accuracy. An upper reference for any 10-image set at K=5.
usage: ceiling2.py --mult 1 --iters 1500 --bs 256 [--seed 42]"""
import argparse, json, math, sys, torch, torch.nn.functional as F
import proto as P
import dd_lib as D

ap = argparse.ArgumentParser()
ap.add_argument('--mult', type=int, default=1)          # 1: 5 steps (n <= 32), 2: 10 steps, 4: 20 steps
ap.add_argument('--iters', type=int, default=1500)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--eval-every', type=int, default=100)
a = ap.parse_args()
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]
R = sum(LRS) * a.mult
m, _, _ = P.fresh_model(a.seed)
theta0 = [p.detach().clone() for p in m.parameters()]
g = torch.Generator().manual_seed(0)
for it in range(a.iters):
    idx = torch.randperm(len(D.xtr), generator=g)[:a.bs]
    xb = P.augment(D.xtr[idx].cuda().permute(0, 3, 1, 2).float().div(255))
    with torch.autocast('cuda', dtype=torch.bfloat16):
        loss = F.cross_entropy(m(xb), D.ytr[idx].cuda())
    m.zero_grad(set_to_none=True); loss.backward()
    step = R * (0.002 + 0.048 * (1 + math.cos(math.pi * it / a.iters)) / 2)
    with torch.no_grad():
        for p, p0 in zip(m.parameters(), theta0):
            p.sub_(step * p.grad.sign())
            p.copy_(torch.max(torch.min(p, p0 + R), p0 - R))
    if (it + 1) % a.eval_every == 0 or it == 0:
        with torch.no_grad():
            frac = sum(((p - p0).abs() > 0.99 * R).sum().item() for p, p0 in zip(m.parameters(), theta0)) / \
                sum(p.numel() for p in m.parameters())
        print(json.dumps({'mult': a.mult, 'R': R, 'it': it + 1, 'loss': round(loss.item(), 4),
                          'val_acc': round(P.evaluate(m, D.xva, D.yva), 4), 'frac_at_box_edge': round(frac, 3)}), flush=True)
