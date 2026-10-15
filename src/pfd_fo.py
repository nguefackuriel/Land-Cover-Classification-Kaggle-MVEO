"""PFD-FO — first-order, memory-constant variant of PFD for any n (any number of official steps).

Pass 1 (no graph): run the exact official schedule on the synthetic set (DataLoader-style shuffle each epoch,
batches of 32 with the leftover batch kept, AdamW, per-epoch cosine lr), with a fresh random head; then
G = d CE(real batch) / d theta_K.
Pass 2: replay each step k with the same batch and augmentation and backpropagate -lr_k * <u_k, G> into the
pixels, where u_k is the straight-through Adam direction of step k (earlier Adam moments detached). This drops
the cross-step terms of the full unroll (pfd.py) but costs one step of memory regardless of K.

usage: pfd_fo.py --init cache/medoids_n64.pt --iters 300 --tag fo_n64
"""
import argparse, json, math, random, sys, time
import torch, torch.nn.functional as F
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel
import dd_lib as D
import proto as P

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--lr', type=float, default=0.03)
ap.add_argument('--tau', type=float, default=0.3)
ap.add_argument('--real-bs', type=int, default=256)
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

idx = torch.load(a.init)
x0, y = D.as_x01(idx)
y = y.to(dev)
n = len(y)
z = torch.logit(x0.clamp(0.02, 0.98)).to(dev).requires_grad_(True)
outer = torch.optim.Adam([z], lr=a.lr)


def schedule():
    """[(lr, batch_indices, rng_state)] exactly like the official DataLoader + per-epoch scheduler."""
    steps = []
    for lr in LRS:
        perm = torch.randperm(n)
        for j in range(0, n, 32):
            steps.append((lr, perm[j:j + 32].to(dev), None))
    return steps


def adam_dir(m, v, g, t):
    """Straight-through Adam direction: exact forward, smoothed-denominator backward."""
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    mh, vh = m / (1 - B1 ** t), v / (1 - B2 ** t)
    exact = mh / (vh.sqrt() + EPS)
    smooth = mh / (vh + (a.tau ** 2) * vh.detach().mean() + EPS ** 2).sqrt()   # finite derivative where v_hat = 0
    return m, v, smooth + (exact - smooth).detach()


def outer_step():
    x01 = torch.sigmoid(z)
    head_w = torch.empty(10, 768, device=dev).uniform_(-HEAD_BOUND, HEAD_BOUND)
    head_b = torch.empty(10, device=dev).uniform_(-HEAD_BOUND, HEAD_BOUND)
    th = dict(theta0)
    th['heads.head.weight'], th['heads.head.bias'] = head_w, head_b
    m = {k: torch.zeros_like(t) for k, t in th.items()}
    v = {k: torch.zeros_like(t) for k, t in th.items()}
    steps, trail = schedule(), []
    # pass 1: exact trajectory, no graph
    for t, (lr, b, _) in enumerate(steps, 1):
        state = random.getstate()
        params = {k: tt.detach().requires_grad_(True) for k, tt in th.items()}
        with torch.no_grad():
            xb = P.augment(x01[b].detach())
        loss = F.cross_entropy(functional_call(base, params, (xb,)), y[b])
        gs = torch.autograd.grad(loss, list(params.values()))
        trail.append((t, lr, b, state))
        with torch.no_grad():
            for (k, p), g in zip(params.items(), gs):
                m[k], v[k], u = adam_dir(m[k], v[k], g, t)
                th[k] = p.detach() * (1 - lr * WD) - lr * u
    # real-data gradient at theta_K
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
    # pass 2: replay the trajectory in order (same batches and crops), push -lr_k <u_k, G> into the pixels,
    # then advance theta/m/v with detached values, so only one step's graph is alive at a time
    outer.zero_grad(set_to_none=True)
    th = {k: theta0[k] for k in theta0}
    th['heads.head.weight'], th['heads.head.bias'] = head_w, head_b
    m = {k: torch.zeros_like(t_) for k, t_ in th.items()}
    v = {k: torch.zeros_like(t_) for k, t_ in th.items()}
    for t, lr, b, state in trail:
        random.setstate(state)
        params = {k: tt.detach().requires_grad_(True) for k, tt in th.items()}
        with sdpa_kernel(SDPBackend.MATH):
            loss = F.cross_entropy(functional_call(base, params, (P.augment(torch.sigmoid(z)[b]),)), y[b])
            gs = torch.autograd.grad(loss, list(params.values()), create_graph=True)
        surrogate, nxt = 0, {}
        for (k, p), g in zip(params.items(), gs):
            m_new, v_new, u = adam_dir(m[k], v[k], g, t)
            surrogate = surrogate - lr * (u * G[k]).sum()
            nxt[k] = (p.detach() * (1 - lr * WD) - lr * u.detach(), m_new.detach(), v_new.detach())
        surrogate.backward()
        th = {k: s[0] for k, s in nxt.items()}
        m = {k: s[1] for k, s in nxt.items()}
        v = {k: s[2] for k, s in nxt.items()}
        del gs, surrogate, params, nxt
    if torch.isfinite(z.grad).all():
        outer.step()
    else:
        print(json.dumps({'skipped': 'non-finite grad'}), flush=True)
    return real_loss, len(steps)


log = open(f'runs/pfd_{a.tag}.jsonl', 'w')
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
                       f'cache/pfd_{a.tag}_best.pt')
    if it == a.iters:
        break
    t0 = time.time()
    real_loss, K = outer_step()
    if it % 10 == 0:
        rec = {'it': it, 'K': K, 'real_loss_after_K': round(real_loss, 4), 'sec': round(time.time() - t0, 2),
               'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
