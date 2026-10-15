"""PFD v1 — Protocol-Faithful Distillation for n <= 32 (5 official AdamW steps, one batch per epoch).

Unrolls the exact official update (AdamW 3e-5, wd .05, per-epoch cosine lr, RRC .75-1 + flips) from the
ImageNet ViT-B/16 with a freshly sampled random head each outer iteration, and backpropagates the loss of the
resulting network on real training images into the synthetic pixels. Adam's early steps are ~sign(g), whose
derivative is ~0, so the update is straight-through: exact Adam forward, Adam with a smoothed denominator backward.

usage: pfd.py --init cache/medoids_ipc1.pt --iters 400 --tag v1
"""
import argparse, json, math, sys, time
import torch, torch.nn.functional as F
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel
import dd_lib as D
import proto as P

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--iters', type=int, default=400)
ap.add_argument('--lr', type=float, default=0.03)          # outer Adam on pixel logits
ap.add_argument('--tau', type=float, default=0.3)          # smoothing of Adam's denominator (backward only)
ap.add_argument('--real-bs', type=int, default=256)
ap.add_argument('--eval-every', type=int, default=50)
ap.add_argument('--tag', default='v1')
ap.add_argument('--heads', type=int, default=1)            # random heads averaged per outer iteration
ap.add_argument('--opt', choices=['adam', 'sgdnorm'], default='adam')  # sgdnorm: z -= lr * g / max|g|
ap.add_argument('--lr-end', type=float, default=None)     # sgdnorm: cosine-decay the step to this value
a = ap.parse_args()

dev = 'cuda'
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]
B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 0.05

base, _, _ = P.fresh_model(0)
base.eval()
names = [n for n, _ in base.named_parameters()]
theta0 = {n: p.detach() for n, p in base.named_parameters()}
for p in base.parameters():
    p.requires_grad_(False)
HEAD_BOUND = 1 / math.sqrt(768)

init = torch.load(a.init)
if isinstance(init, dict):          # resume from a previous run's best images
    x0, y = init['x01'], init['y']
else:                               # train-set indices of a coreset
    x0, y = D.as_x01(init)
y = y.to(dev)
z = torch.logit(x0.clamp(0.02, 0.98)).to(dev).requires_grad_(True)
outer = torch.optim.Adam([z], lr=a.lr)


def sample_head():
    w = torch.empty(10, 768, device=dev).uniform_(-HEAD_BOUND, HEAD_BOUND)
    b = torch.empty(10, device=dev).uniform_(-HEAD_BOUND, HEAD_BOUND)
    return w, b


def unrolled(x01):
    """Return the parameter dict after the 5 official steps, differentiable w.r.t. x01."""
    w, b = sample_head()
    th = dict(theta0)
    th['heads.head.weight'], th['heads.head.bias'] = w, b
    m = {k: torch.zeros_like(v) for k, v in th.items()}
    v = {k: torch.zeros_like(t) for k, t in th.items()}
    for t, lr in enumerate(LRS, 1):
        params = {k: th[k] if th[k].requires_grad else th[k].detach().requires_grad_(True) for k in th}
        with sdpa_kernel(SDPBackend.MATH):
            loss = F.cross_entropy(functional_call(base, params, (P.augment(x01),)), y)
            grads = torch.autograd.grad(loss, list(params.values()), create_graph=True)
        new = {}
        for (k, p), g in zip(params.items(), grads):
            m[k] = B1 * m[k] + (1 - B1) * g
            v[k] = B2 * v[k] + (1 - B2) * g * g
            mh, vh = m[k] / (1 - B1 ** t), v[k] / (1 - B2 ** t)
            exact = mh / (vh.sqrt() + EPS)
            delta2 = (a.tau ** 2) * vh.detach().mean() + EPS ** 2
            smooth = mh / (vh + delta2).sqrt()   # finite derivative where v_hat = 0
            upd = smooth + (exact - smooth).detach()
            new[k] = p * (1 - lr * WD) - lr * upd
        th = new
    return th


def real_grad(th):
    """d CE(real batch) / d theta_K, computed without a graph (large batch, chunked)."""
    det = {k: t.detach().requires_grad_(True) for k, t in th.items()}
    ridx = torch.randperm(len(D.xtr))[:a.real_bs]
    total = 0.0
    acc = {k: torch.zeros_like(t) for k, t in det.items()}
    for j in range(0, a.real_bs, 64):
        xb = P.to_eval(D.xtr[ridx[j:j + 64]].to(dev))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = F.cross_entropy(functional_call(base, det, (xb,)), D.ytr[ridx[j:j + 64]].to(dev)) * len(xb) / a.real_bs
        gs = torch.autograd.grad(l, list(det.values()))
        for (k, _), g in zip(det.items(), gs):
            acc[k] += g.float()
        total += l.item()
    return acc, total


def quantized():
    return (torch.sigmoid(z.detach()) * 255).round().div(255).cpu()


log = open(f'runs/pfd_{a.tag}.jsonl', 'w')
best = (-1, None)
for it in range(a.iters + 1):
    if it % a.eval_every == 0:
        xq = quantized()
        accs = [round(D.train_eval(xq, y.cpu(), s), 4) for s in (42, 0, 1)]
        rec = {'it': it, 'val_accs': accs, 'mean': round(sum(accs) / 3, 4)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        if rec['mean'] > best[0]:
            best = (rec['mean'], xq.clone())
            torch.save({'x01': xq, 'y': y.cpu(), 'it': it, 'accs': accs}, f'cache/pfd_{a.tag}_best.pt')
    if it == a.iters:
        break
    t0 = time.time()
    outer.zero_grad(set_to_none=True)
    real_loss = 0.0
    for _ in range(a.heads):                 # average the meta-gradient over several random heads
        th = unrolled(torch.sigmoid(z))
        G, rl = real_grad(th)
        keys = [k for k in th if th[k].requires_grad]
        torch.autograd.backward([th[k] for k in keys], [G[k] / a.heads for k in keys])
        real_loss += rl / a.heads
        del th, G
    if torch.isfinite(z.grad).all():
        if a.opt == 'adam':
            outer.step()
        else:   # the meta-gradient only descends for small steps (diag_metagrad.py), so keep its shape
            with torch.no_grad():
                lr_end = a.lr if a.lr_end is None else a.lr_end
                step = lr_end + (a.lr - lr_end) * (1 + math.cos(math.pi * it / a.iters)) / 2
                z -= step * z.grad / z.grad.abs().max().clamp_min(1e-12)
    else:
        print(json.dumps({'it': it, 'skipped': 'non-finite grad'}), flush=True)
    if it % 10 == 0:
        rec = {'it': it, 'real_loss_after_5_steps': round(real_loss, 4), 'sec': round(time.time() - t0, 2),
               'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
