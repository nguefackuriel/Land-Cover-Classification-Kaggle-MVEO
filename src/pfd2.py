"""PFD2 — protocol-faithful distillation for any n (any number of official steps K = 5*ceil(n/32)).

Forward: the exact official schedule (DataLoader-style shuffle each epoch, batches of 32 with the leftover batch
kept, AdamW 3e-5 / wd .05, per-epoch cosine lr) from the ImageNet ViT-B/16 with a fresh random head.
Backward: full straight-through unroll over the first `--window` steps (where the large learning rates are);
the remaining steps are run exactly but treated as identity in the backward pass (their own updates are
constants w.r.t. the pixels). With --window >= K this is pfd.py's full unroll.
Outer update: normalized descent z -= eps * g / max|g| (the meta-gradient only descends for small steps).

usage: pfd2.py --init cache/medoids_n64.pt --window 6 --iters 300 --tag w6_n64
"""
import argparse, json, math, sys, time
import torch, torch.nn.functional as F
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel
import dd_lib as D
import proto as P

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--window', type=int, default=6)
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--lr-end', type=float, default=0.01)
ap.add_argument('--tau', type=float, default=0.3)
ap.add_argument('--heads', type=int, default=2)
ap.add_argument('--real-bs', type=int, default=256)
ap.add_argument('--bf16-inner', action='store_true')
ap.add_argument('--eval-every', type=int, default=50)
ap.add_argument('--tag', required=True)
a = ap.parse_args()

dev = 'cuda'
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]
B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 0.05
HEAD_BOUND = 1 / math.sqrt(768)

base, _, _ = P.fresh_model(0)
base.eval()
for p in base.parameters():
    p.requires_grad_(False)
theta0 = {k: p.detach() for k, p in base.named_parameters()}

init = torch.load(a.init)
x0, y = (init['x01'], init['y']) if isinstance(init, dict) else D.as_x01(init)
y = y.to(dev)
n = len(y)
z = torch.logit(x0.clamp(0.02, 0.98)).to(dev).requires_grad_(True)


def adam_dir(m, v, g, t):
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    mh, vh = m / (1 - B1 ** t), v / (1 - B2 ** t)
    exact = mh / (vh.sqrt() + EPS)
    smooth = mh / (vh + a.tau ** 2 * vh.detach().mean() + EPS ** 2).sqrt()
    return m, v, smooth + (exact - smooth).detach()


def inner_loss(params, xb, yb):
    if a.bf16_inner:
        with torch.autocast('cuda', dtype=torch.bfloat16):
            return F.cross_entropy(functional_call(base, params, (xb,)).float(), yb)
    return F.cross_entropy(functional_call(base, params, (xb,)), yb)


def one_head():
    """Accumulate d CE_real(theta_K) / d z for one random head into z.grad; return the real loss."""
    x01 = torch.sigmoid(z)
    th = dict(theta0)
    th['heads.head.weight'] = torch.empty(10, 768, device=dev).uniform_(-HEAD_BOUND, HEAD_BOUND)
    th['heads.head.bias'] = torch.empty(10, device=dev).uniform_(-HEAD_BOUND, HEAD_BOUND)
    m = {k: torch.zeros_like(t) for k, t in th.items()}
    v = {k: torch.zeros_like(t) for k, t in th.items()}
    t, th_window = 0, None
    for lr in LRS:
        perm = torch.randperm(n, device=dev)
        for j in range(0, n, 32):
            b = perm[j:j + 32]
            t += 1
            graph = t <= a.window
            if graph:
                params = {k: tt if tt.requires_grad else tt.detach().requires_grad_(True) for k, tt in th.items()}
                with sdpa_kernel(SDPBackend.MATH):
                    gs = torch.autograd.grad(inner_loss(params, P.augment(x01[b]), y[b]), list(params.values()),
                                             create_graph=True)
                new = {}
                for (k, p), g in zip(params.items(), gs):
                    m[k], v[k], u = adam_dir(m[k], v[k], g, t)
                    new[k] = p * (1 - lr * WD) - lr * u
                th = new
                if t == a.window:
                    th_window = th
                    th = {k: tt.detach() for k, tt in th.items()}
                    m = {k: tt.detach() for k, tt in m.items()}
                    v = {k: tt.detach() for k, tt in v.items()}
            else:
                params = {k: tt.detach().requires_grad_(True) for k, tt in th.items()}
                with torch.no_grad():
                    xb = P.augment(x01[b].detach())
                gs = torch.autograd.grad(inner_loss(params, xb, y[b]), list(params.values()))
                with torch.no_grad():
                    for (k, p), g in zip(params.items(), gs):
                        m[k], v[k], u = adam_dir(m[k], v[k], g, t)
                        th[k] = p.detach() * (1 - lr * WD) - lr * u
    if th_window is None:          # window >= K: the whole trajectory carries the graph
        th_window, th = th, {k: tt.detach() for k, tt in th.items()}
    det = {k: tt.detach().requires_grad_(True) for k, tt in th.items()}
    ridx = torch.randperm(len(D.xtr))[:a.real_bs]
    G = {k: torch.zeros_like(tt) for k, tt in det.items()}; real_loss = 0.0
    for j in range(0, a.real_bs, 64):
        xb = P.to_eval(D.xtr[ridx[j:j + 64]].to(dev))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = F.cross_entropy(functional_call(base, det, (xb,)), D.ytr[ridx[j:j + 64]].to(dev)) * len(xb) / a.real_bs
        for (k, _), g in zip(det.items(), torch.autograd.grad(l, list(det.values()))):
            G[k] += g.float()
        real_loss += l.item()
    keys = [k for k in th_window if th_window[k].requires_grad]
    torch.autograd.backward([th_window[k] for k in keys], [G[k] / a.heads for k in keys])
    return real_loss, t


log = open(f'runs/pfd2_{a.tag}.jsonl', 'w')
best = -1
for it in range(a.iters + 1):
    if it % a.eval_every == 0:
        xq = (torch.sigmoid(z.detach()) * 255).round().div(255).cpu()
        accs = [round(D.train_eval(xq, y.cpu(), s), 4) for s in (42, 0, 1)]
        mean = sum(accs) / 3
        rec = {'it': it, 'n': n, 'val_accs': accs, 'mean': round(mean, 4), 'score_mean': round(D.score(mean, n), 4)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if mean > best:
            best = mean
            torch.save({'x01': xq, 'y': y.cpu(), 'it': it, 'accs': accs},
                       f'cache/pfd2_{a.tag}_best.pt')
    if it == a.iters:
        break
    t0 = time.time()
    z.grad = None
    real_loss = 0.0
    for _ in range(a.heads):
        rl, K = one_head()
        real_loss += rl / a.heads
    if torch.isfinite(z.grad).all():
        eps = a.lr_end + (a.lr - a.lr_end) * (1 + math.cos(math.pi * it / a.iters)) / 2
        with torch.no_grad():
            z -= eps * z.grad / z.grad.abs().max().clamp_min(1e-12)
    else:
        print(json.dumps({'it': it, 'skipped': 'non-finite grad'}), flush=True)
    if it % 10 == 0:
        rec = {'it': it, 'K': K, 'window': min(a.window, K), 'real_loss_after_K': round(real_loss, 4),
               'sec': round(time.time() - t0, 2), 'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
