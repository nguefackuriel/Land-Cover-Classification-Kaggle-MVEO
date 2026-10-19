"""Corner descent: the exact official run ends near the corner theta0 + R * sign(update) of its step gradients. Model it
smoothly as C(x) = theta0 - R * tanh(gbar / (tau * rms)), gbar = sum_k (lr_k / R) g_k, with each g_k taken at the real
run's own state theta_k (recorded views, seed-42 head, no graph through earlier steps). Descend the real calibrated CE
of C(x) with respect to the pixels: one double-backward per step, no chaotic unroll, no fixed target.
usage: PROTO_PIL=1 cornerdesc.py --init set.pt --views views.pt --tag cd1"""
import argparse, json, math, sys, time
import torch, torch.nn.functional as F
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel
import dd_lib as D
import proto as P

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--views', required=True)
ap.add_argument('--iters', type=int, default=200)
ap.add_argument('--lr', type=float, default=0.5)        # Adam on pixels, 1/255 levels
ap.add_argument('--lr-end', type=float, default=0.05)
ap.add_argument('--warmup', type=int, default=10)
ap.add_argument('--tau', type=float, default=0.3)
ap.add_argument('--real-bs', type=int, default=1024)
ap.add_argument('--eval-every', type=int, default=20)
ap.add_argument('--tag', default='cd')
a = ap.parse_args()
dev = 'cuda'
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]
R = sum(LRS); WD = 0.05; B1, B2, EPS = 0.9, 0.999, 1e-8
base, _, _ = P.fresh_model(42)
base.eval()
for p in base.parameters():
    p.requires_grad_(False)
th0 = {k: p.detach().clone() for k, p in base.named_parameters()}
V = torch.load(a.views)
x0, y = D.load_set(a.init)
z = x0.clone().to(dev).requires_grad_(True)
y = y.to(dev)
mo, vo = torch.zeros_like(z), torch.zeros_like(z)


@torch.no_grad()
def actual_path(x):
    th = dict(th0); m = {k: torch.zeros_like(t) for k, t in th.items()}; v = {k: torch.zeros_like(t) for k, t in th.items()}
    out = []
    for e, lr in enumerate(LRS):
        out.append(th)
        (idx, vw), = V['epochs'][e]
        params = {k: t.detach().requires_grad_(True) for k, t in th.items()}
        with torch.enable_grad():
            ce = F.cross_entropy(functional_call(base, params, (P.augment_views(x[idx.to(dev)], vw),)).float(), y[idx.to(dev)])
            gs = torch.autograd.grad(ce, list(params.values()))
        t = e + 1; nth = {}
        for (k, p), g in zip(params.items(), gs):
            m[k] = B1 * m[k] + (1 - B1) * g; v[k] = B2 * v[k] + (1 - B2) * g * g
            nth[k] = p.detach() * (1 - lr * WD) - lr * (m[k] / (1 - B1 ** t)) / ((v[k] / (1 - B2 ** t)).sqrt() + EPS)
        th = nth
    return out


def step_grad(x, e, th, create):
    (idx, vw), = V['epochs'][e]
    params = {k: t.detach().requires_grad_(True) for k, t in th.items()}
    with sdpa_kernel(SDPBackend.MATH):
        ce = F.cross_entropy(functional_call(base, params, (P.augment_views(x[idx.to(dev)], vw),)).float(), y[idx.to(dev)])
    return list(params.keys()), torch.autograd.grad(ce, list(params.values()), create_graph=create)


def real_grad(th):
    ridx = torch.randperm(len(D.xtr))[:a.real_bs]
    det = {k: t.detach().requires_grad_(True) for k, t in th.items()}
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = torch.cat([functional_call(base, det, (P.to_eval(D.xtr[ridx[j:j + 256]].to(dev)),)).float() for j in range(0, a.real_bs, 256)])
    yb = D.ytr[ridx].to(dev)
    Ts = torch.logspace(-2, 1, 61, device=dev)
    T = Ts[torch.stack([F.cross_entropy(o / t, yb) for t in Ts]).argmin()].item()
    G = {k: torch.zeros_like(t) for k, t in det.items()}; tot = 0.
    for j in range(0, a.real_bs, 128):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = F.cross_entropy(functional_call(base, det, (P.to_eval(D.xtr[ridx[j:j + 128]].to(dev)),)).float() / T, yb[j:j + 128]) * 128 / a.real_bs
        for (k, _), g in zip(det.items(), torch.autograd.grad(l, list(det.values()))):
            G[k] += g.float()
        tot += l.item()
    return G, tot, T


log = open(f'runs/cornerdesc_{a.tag}.jsonl', 'w')
best = -1
for it in range(a.iters + 1):
    if it % a.eval_every == 0:
        xq = (z.detach().clamp(0, 1) * 255).round().div(255).cpu()
        ex = D.train_eval_views(xq, y.cpu(), 42, V)[-1]
        rec = {'it': it, 'exact_acc': round(ex, 4)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        torch.save({'x01': xq, 'y': y.cpu(), 'it': it}, f'cache/cd_{a.tag}_it{it}.pt')
        if ex > best:
            best = ex
            torch.save({'x01': xq, 'y': y.cpu(), 'it': it}, f'cache/cd_{a.tag}_best.pt')
    if it == a.iters:
        break
    t0 = time.time()
    x = z.clamp(0, 1)
    states = actual_path(x.detach())
    # gbar without graph, to build the smooth corner and its real-loss gradient
    gbar = None
    for e in range(5):
        keys, gs = step_grad(x.detach(), e, states[e], False)
        if gbar is None:
            gbar = {k: torch.zeros_like(g) for k, g in zip(keys, gs)}
        for k, g in zip(keys, gs):
            gbar[k] += LRS[e] / R * g.float()
    sig = {k: a.tau * g.pow(2).mean().sqrt().clamp_min(1e-20) for k, g in gbar.items()}
    corner = {k: th0[k] - R * torch.tanh(gbar[k] / sig[k]) for k in gbar}
    G, rl, T = real_grad(corner)
    # dL/dgbar_k = G * (-R / sig) * sech^2(gbar / sig); push it back through each step's gradient graph
    C = {k: G[k] * (-R / sig[k]) * (1 - torch.tanh(gbar[k] / sig[k]) ** 2) for k in gbar}
    del corner, G
    gz = torch.zeros_like(z)
    for e in range(5):
        keys, gs = step_grad(z.clamp(0, 1), e, states[e], True)
        s = sum((LRS[e] / R * C[k] * g.float()).sum() for k, g in zip(keys, gs))
        gz += torch.autograd.grad(s, z, retain_graph=False)[0]
        del gs, s
    del states, gbar, C
    eps = (a.lr_end + (a.lr - a.lr_end) * (1 + math.cos(math.pi * it / a.iters)) / 2) * min(1., (it + 1) / max(a.warmup, 1))
    with torch.no_grad():
        mo.mul_(0.9).add_(gz, alpha=0.1); vo.mul_(0.99).addcmul_(gz, gz, value=0.01)
        vh = vo / (1 - 0.99 ** (it + 1))
        z -= eps / 255 * (mo / (1 - 0.9 ** (it + 1))) / (vh.sqrt() + 1e-8 * vh.sqrt().mean())
        z.clamp_(0, 1)
    if it % 5 == 0:
        rec = {'it': it, 'corner_ce': round(rl, 4), 'T': round(T, 3), 'sec': round(time.time() - t0, 1),
               'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
