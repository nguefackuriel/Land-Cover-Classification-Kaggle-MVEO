"""Is the PFD meta-gradient a descent direction? With head, crops and real batch frozen, step the pixel logits
along -d for several step sizes and measure the exact real loss after the official 5 steps (exact AdamW).
Compares the full-unroll straight-through gradient, the first-order replay gradient, and a random direction."""
import math, random, sys, torch, torch.nn.functional as F
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel
import dd_lib as D
import proto as P

dev = 'cuda'
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]
B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 0.05
TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 0.3

base, _, _ = P.fresh_model(0); base.eval()
for p in base.parameters():
    p.requires_grad_(False)
theta0 = {k: p.detach() for k, p in base.named_parameters()}
idx = torch.load('cache/medoids_ipc1.pt')
x0, y = D.as_x01(idx); y = y.to(dev)
z0 = torch.logit(x0.clamp(0.02, 0.98)).to(dev)
torch.manual_seed(7)
B = 1 / math.sqrt(768)
HW = torch.empty(10, 768, device=dev).uniform_(-B, B); HB = torch.empty(10, device=dev).uniform_(-B, B)
RIDX = torch.randperm(len(D.xtr))[:1024]
XR = torch.cat([P.to_eval(D.xtr[RIDX[j:j + 128]].to(dev)) for j in range(0, 1024, 128)])
YR = D.ytr[RIDX].to(dev)


def adam(m, v, g, t, smooth_bwd):
    m = B1 * m + (1 - B1) * g; v = B2 * v + (1 - B2) * g * g
    mh, vh = m / (1 - B1 ** t), v / (1 - B2 ** t)
    exact = mh / (vh.sqrt() + EPS)
    if not smooth_bwd:
        return m, v, exact
    smooth = mh / (vh + TAU ** 2 * vh.detach().mean() + EPS ** 2).sqrt()
    return m, v, smooth + (exact - smooth).detach()


def init_state():
    th = dict(theta0); th['heads.head.weight'], th['heads.head.bias'] = HW, HB
    return th, {k: torch.zeros_like(t) for k, t in th.items()}, {k: torch.zeros_like(t) for k, t in th.items()}


def real_loss_and_grad(th, need_grad):
    det = {k: t.detach().requires_grad_(need_grad) for k, t in th.items()}
    G = {k: torch.zeros_like(t) for k, t in det.items()}; tot = 0.0
    for j in range(0, 1024, 128):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = F.cross_entropy(functional_call(base, det, (XR[j:j + 128],)), YR[j:j + 128]) * 128 / 1024
        if need_grad:
            for (k, _), g in zip(det.items(), torch.autograd.grad(l, list(det.values()))):
                G[k] += g.float()
        tot += l.item()
    return tot, G


def exact_loss(z):
    random.seed(1234); th, m, v = init_state()
    for t, lr in enumerate(LRS, 1):
        params = {k: tt.detach().requires_grad_(True) for k, tt in th.items()}
        loss = F.cross_entropy(functional_call(base, params, (P.augment(torch.sigmoid(z)),)), y)
        gs = torch.autograd.grad(loss, list(params.values()))
        with torch.no_grad():
            for (k, p), g in zip(params.items(), gs):
                m[k], v[k], u = adam(m[k], v[k], g, t, False)
                th[k] = p * (1 - lr * WD) - lr * u
    return real_loss_and_grad(th, False)[0]


def grad_full(z0):
    z = z0.clone().requires_grad_(True); random.seed(1234); th, m, v = init_state()
    for t, lr in enumerate(LRS, 1):
        params = {k: tt if tt.requires_grad else tt.detach().requires_grad_(True) for k, tt in th.items()}
        with sdpa_kernel(SDPBackend.MATH):
            loss = F.cross_entropy(functional_call(base, params, (P.augment(torch.sigmoid(z)),)), y)
            gs = torch.autograd.grad(loss, list(params.values()), create_graph=True)
        new = {}
        for (k, p), g in zip(params.items(), gs):
            m[k], v[k], u = adam(m[k], v[k], g, t, True)
            new[k] = p * (1 - lr * WD) - lr * u
        th = new
    _, G = real_loss_and_grad(th, True)
    keys = list(th)
    torch.autograd.backward([th[k] for k in keys], [G[k] for k in keys])
    return z.grad.detach()


def grad_fo(z0):
    # pass 1 at exact trajectory to get theta_K, then replay with graph
    random.seed(1234); th, m, v = init_state()
    for t, lr in enumerate(LRS, 1):
        params = {k: tt.detach().requires_grad_(True) for k, tt in th.items()}
        loss = F.cross_entropy(functional_call(base, params, (P.augment(torch.sigmoid(z0)),)), y)
        gs = torch.autograd.grad(loss, list(params.values()))
        with torch.no_grad():
            for (k, p), g in zip(params.items(), gs):
                m[k], v[k], u = adam(m[k], v[k], g, t, False)
                th[k] = p * (1 - lr * WD) - lr * u
    _, G = real_loss_and_grad(th, True)
    z = z0.clone().requires_grad_(True); random.seed(1234); th, m, v = init_state()
    for t, lr in enumerate(LRS, 1):
        params = {k: tt.detach().requires_grad_(True) for k, tt in th.items()}
        with sdpa_kernel(SDPBackend.MATH):
            loss = F.cross_entropy(functional_call(base, params, (P.augment(torch.sigmoid(z)),)), y)
            gs = torch.autograd.grad(loss, list(params.values()), create_graph=True)
        sur, nxt = 0, {}
        for (k, p), g in zip(params.items(), gs):
            mn, vn, u = adam(m[k], v[k], g, t, True)
            sur = sur - lr * (u * G[k]).sum()
            nxt[k] = (p.detach() * (1 - lr * WD) - lr * u.detach(), mn.detach(), vn.detach())
        sur.backward()
        th = {k: s[0] for k, s in nxt.items()}; m = {k: s[1] for k, s in nxt.items()}; v = {k: s[2] for k, s in nxt.items()}
    return z.grad.detach()


def grad_window(z0, W):
    """Full straight-through unroll over the first W steps, exact no-graph steps afterwards (identity backward)."""
    z = z0.clone().requires_grad_(True); random.seed(1234); th, m, v = init_state(); thW = None
    for t, lr in enumerate(LRS, 1):
        if t <= W:
            params = {k: tt if tt.requires_grad else tt.detach().requires_grad_(True) for k, tt in th.items()}
            with sdpa_kernel(SDPBackend.MATH):
                loss = F.cross_entropy(functional_call(base, params, (P.augment(torch.sigmoid(z)),)), y)
                gs = torch.autograd.grad(loss, list(params.values()), create_graph=True)
            new = {}
            for (k, p), g in zip(params.items(), gs):
                m[k], v[k], u = adam(m[k], v[k], g, t, True)
                new[k] = p * (1 - lr * WD) - lr * u
            th = new
            if t == W:
                thW = th
                th = {k: q.detach() for k, q in th.items()}; m = {k: q.detach() for k, q in m.items()}; v = {k: q.detach() for k, q in v.items()}
        else:
            params = {k: tt.detach().requires_grad_(True) for k, tt in th.items()}
            loss = F.cross_entropy(functional_call(base, params, (P.augment(torch.sigmoid(z).detach()),)), y)
            gs = torch.autograd.grad(loss, list(params.values()))
            with torch.no_grad():
                for (k, p), g in zip(params.items(), gs):
                    m[k], v[k], u = adam(m[k], v[k], g, t, False)
                    th[k] = p * (1 - lr * WD) - lr * u
    _, G = real_loss_and_grad(th, True)
    keys = list(thW)
    torch.autograd.backward([thW[k] for k in keys], [G[k] for k in keys])
    return z.grad.detach()


L0 = exact_loss(z0)
print(f'tau={TAU} base real loss after 5 steps: {L0:.4f} (re-run noise: {exact_loss(z0) - L0:+.5f})', flush=True)
dirs = {'full_unroll': grad_full(z0)}
for W in (1, 2, 3):
    dirs[f'window_{W}'] = grad_window(z0, W)
if len(sys.argv) <= 2:
    dirs.update({'first_order': grad_fo(z0), 'random': torch.randn_like(z0)})
for name, d in dirs.items():
    if name != 'full_unroll':
        c = F.cosine_similarity(dirs['full_unroll'].flatten(), d.flatten(), dim=0).item()
        print(f'cos(full, {name}) = {c:.3f}', flush=True)
for name, d in dirs.items():
    d = d / d.abs().max()
    out = []
    for eps in [0.05, 0.2, 0.5, 1.0]:
        out.append(f'{eps}:{exact_loss(z0 - eps * d) - L0:+.4f}')
    print(f'{name:12s} dL at step size ' + '  '.join(out), flush=True)
