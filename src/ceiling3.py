"""Tighter estimate of the best network reachable in K official Adam steps (seed-42 head). Like ceiling2.py (L-inf box
of radius sum(lr) around the ImageNet weights), but: real train images seen through the *eval* transform (what val/test
use), outer CE at a low or fitted temperature (accuracy, not logit scale), projected Adam or sign steps, more iters.
usage: ceiling3.py --T 0 --opt adam --iters 1500 --bs 256"""
import argparse, json, math, sys, torch, torch.nn.functional as F
import proto as P
import dd_lib as D

ap = argparse.ArgumentParser()
ap.add_argument('--mult', type=float, default=1)
ap.add_argument('--iters', type=int, default=1500)
ap.add_argument('--bs', type=int, default=256)
ap.add_argument('--micro', type=int, default=128)
ap.add_argument('--T', type=float, default=0.)          # 0: temperature fitted on each batch
ap.add_argument('--opt', default='adam', choices=['adam', 'sign'])
ap.add_argument('--lr', type=float, default=0.05)        # in units of R
ap.add_argument('--aug', action='store_true')            # train views instead of eval transform
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--eval-every', type=int, default=100)
ap.add_argument('--tag', default='c3')
ap.add_argument('--init-state', default='')   # start inside the box from this state dict (e.g. the end of our exact run)
ap.add_argument('--save-every', type=int, default=0)
ap.add_argument('--data-seed', type=int, default=0)
a = ap.parse_args()
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]
R = sum(LRS) * a.mult
m, _, _ = P.fresh_model(a.seed)
params = list(m.parameters())
theta0 = [p.detach().clone() for p in params]
if a.init_state:
    _sd = torch.load(a.init_state)
    with torch.no_grad():
        for (k, p), p0 in zip(m.named_parameters(), theta0):
            p.copy_(torch.max(torch.min(_sd[k].cuda(), p0 + R), p0 - R))
    print(json.dumps({'tag': a.tag, 'init_val': round(P.evaluate(m, D.xva, D.yva), 4)}), flush=True)
mo = [torch.zeros_like(p) for p in params]; vo = [torch.zeros_like(p) for p in params]
g = torch.Generator().manual_seed(a.data_seed)
Ts = torch.logspace(-2, 1, 61, device='cuda')
for it in range(a.iters):
    idx = D.sample_train(a.bs, g)
    m.zero_grad(set_to_none=True)
    if a.T <= 0:
        with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
            m.eval()
            o = torch.cat([m(P.to_eval(D.xtr[idx[j:j + a.micro]].cuda())).float() for j in range(0, a.bs, a.micro)])
        yb = D.ytr[idx].cuda()
        T = Ts[torch.stack([F.cross_entropy(o / t, yb) for t in Ts]).argmin()].item()
    else:
        T = a.T
    m.eval()      # dropout is 0 in torchvision ViT anyway
    tot = 0.
    for j in range(0, a.bs, a.micro):
        xu = D.xtr[idx[j:j + a.micro]].cuda()
        xb = P.augment(xu.permute(0, 3, 1, 2).float().div(255)) if a.aug else P.to_eval(xu)
        with torch.autocast('cuda', dtype=torch.bfloat16):
            loss = F.cross_entropy(m(xb).float() / T, D.ytr[idx[j:j + a.micro]].cuda()) * len(xu) / a.bs
        loss.backward(); tot += loss.item()
    frac = it / a.iters
    step = R * a.lr * (0.04 + 0.96 * (1 + math.cos(math.pi * frac)) / 2)
    with torch.no_grad():
        for p, p0, mm, vv in zip(params, theta0, mo, vo):
            gr = p.grad.float()
            if a.opt == 'sign':
                p.sub_(step * gr.sign())
            else:
                mm.mul_(0.9).add_(gr, alpha=0.1); vv.mul_(0.99).addcmul_(gr, gr, value=0.01)
                mh = mm / (1 - 0.9 ** (it + 1)); vh = vv / (1 - 0.99 ** (it + 1))
                p.sub_(step * mh / (vh.sqrt() + 1e-12))
            p.copy_(torch.max(torch.min(p, p0 + R), p0 - R))
    if (it + 1) % a.eval_every == 0 or it == 0:
        with torch.no_grad():
            edge = sum(((p - p0).abs() > 0.99 * R).sum().item() for p, p0 in zip(params, theta0)) / \
                sum(p.numel() for p in params)
        print(json.dumps({'tag': a.tag, 'it': it + 1, 'T': round(T, 3), 'loss': round(tot, 4),
                          'val_acc': round(P.evaluate(m, D.xva, D.yva), 4), 'edge': round(edge, 3)}), flush=True)
    if a.save_every and (it + 1) % a.save_every == 0:
        torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, f'cache/ceiling3_{a.tag}_it{it + 1}.pt')
torch.save({k: v.detach().cpu() for k, v in m.state_dict().items()}, f'cache/ceiling3_{a.tag}.pt')
