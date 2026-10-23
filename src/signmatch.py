"""Sign-matching distillation for the exact official run. Early AdamW steps are sign steps, and the corner
theta0 + R*sign(theta* - theta0) of a good reachable network theta* keeps its accuracy (88.3% vs 88.1%). So if the batch
gradient at every official step has sign -s (s = sign(theta* - theta0)), the run ends at that corner. The path is then
known in advance (theta_k = theta_{k-1}(1 - lr wd) + lr_k s), so each step k is scored at its own target point with its
recorded view: no unroll and no chaos. Loss = mean over weights of softplus(g_k * s / (tau * rms_tensor(g_k))).
usage: PROTO_PIL=1 signmatch.py --init cache/pfd3_n10_exact_s2_it280.pt --views cache/views_n10_s42.pt \
       --target cache/ceiling3_near_s2.pt --tag sm1"""
import argparse, json, math, sys, time
import torch, torch.nn.functional as F
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel
import dd_lib as D
import proto as P

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--views', required=True)
ap.add_argument('--target', required=True)
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--lr-end', type=float, default=0.005)
ap.add_argument('--tau', type=float, default=0.3)
ap.add_argument('--steps', default='0,1,2,3,4')
ap.add_argument('--wlr', action='store_true')          # weight step k by lr_k
ap.add_argument('--bf16', action='store_true')
ap.add_argument('--eval-every', type=int, default=20)
ap.add_argument('--tag', default='sm')
ap.add_argument('--warmup', type=int, default=0)
ap.add_argument('--loss', default='softplus', choices=['softplus', 'sigmoid'])   # sigmoid: smooth agreement count
ap.add_argument('--pfd-w', type=float, default=0.)      # >0: add the exact PFD meta-gradient (real loss), both RMS-normalised
ap.add_argument('--dither', type=float, default=0.)   # adam: start each pixel at a random point of its 8-bit bin (width in levels, <1)
ap.add_argument('--dither-seed', type=int, default=0)
ap.add_argument('--disp', action='store_true')   # loss on each weight's final AdamW displacement (momentum included), not per step
ap.add_argument('--tau-d', type=float, default=0.25)   # --disp: softplus(-q / tau_d), q = s * displacement / R in [-1, 1]
ap.add_argument('--pfd-tmult', type=float, default=1.)   # PFD term: real CE at the fitted temperature times this
ap.add_argument('--real-bs', type=int, default=512)        # linear lr warmup over this many iterations
ap.add_argument('--path', default='target', choices=['target', 'actual'])  # actual: score each step where the real run is
ap.add_argument('--opt', default='maxnorm', choices=['maxnorm', 'adam'])   # adam: pixels directly, lr in 1/255 levels
ap.add_argument('--weight', default='uniform', choices=['uniform', 'grad', 'delta'])  # grad: |dCE_T/dtheta| at the corner; delta: |theta*-theta0|/R
a = ap.parse_args()
dev = 'cuda'
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]
WD = 0.05
base, _, _ = P.fresh_model(42)
base.eval()
for p in base.parameters():
    p.requires_grad_(False)
th0 = {k: p.detach().clone() for k, p in base.named_parameters()}     # seed-42 head included
TG = [torch.load(t) for t in a.target.split(',')]     # several targets: consensus sign, weight = |mean sign|
S = {k: sum(torch.sign(t[k].to(dev) - th0[k]) for t in TG) / len(TG) for k in th0}
CW = {k: s_.abs() for k, s_ in S.items()} if len(TG) > 1 else None
S = {k: torch.sign(s_) for k, s_ in S.items()}
del TG
# target path: the official AdamW step with u = -s exactly
path, th = [], dict(th0)
for lr in LRS:
    path.append(th)
    th = {k: t * (1 - lr * WD) + lr * S[k] for k, t in th.items()}
V = torch.load(a.views)
STEPS = [int(s) for s in a.steps.split(',')]
x0, y = D.load_set(a.init)
x0, y = x0.to(dev), y.to(dev)
ADAM = a.opt == 'adam'
z = (x0.clone() if ADAM else torch.logit(x0.clamp(0.02, 0.98))).requires_grad_(True)
if ADAM and a.dither > 0:   # same PNG at the start, but pixels no longer cross a rounding edge all at the same iteration
    with torch.no_grad():
        _r = torch.rand(z.shape, generator=torch.Generator().manual_seed(a.dither_seed)).to(dev)
        z += (_r - 0.5) * a.dither / 255
        z.clamp_(0, 1)
pix = (lambda t: t.clamp(0, 1)) if ADAM else torch.sigmoid
mo, vo = torch.zeros_like(z), torch.zeros_like(z)
NT = sum(t.numel() for t in th0.values())
W = None
if a.weight == 'delta':     # how decided the target is about each weight's sign
    _t = torch.load(a.target.split(',')[0])
    W = {k: ((_t[k].to(dev).float() - th0[k]).abs() / sum(LRS)).clamp(max=1.) for k in th0}
    _m = sum(w.sum().item() for w in W.values()) / NT
    W = {k: w / _m for k, w in W.items()}
    del _t
if a.weight == 'grad':      # how much the real loss cares about each weight at the end of the target path
    corner = {k: t * (1 - LRS[-1] * WD) + LRS[-1] * S[k] for k, t in path[-1].items()}
    det = {k: t.detach().requires_grad_(True) for k, t in corner.items()}
    ridx = D.sample_train(2048, torch.Generator().manual_seed(9))
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = torch.cat([functional_call(base, corner, (P.to_eval(D.xtr[ridx[j:j + 256]].to(dev)),)).float() for j in range(0, 2048, 256)])
    yb = D.ytr[ridx].to(dev)
    Ts = torch.logspace(-2, 1, 61, device=dev)
    T = Ts[torch.stack([F.cross_entropy(o / t, yb) for t in Ts]).argmin()].item()
    W = {k: torch.zeros_like(t) for k, t in det.items()}
    for j in range(0, 2048, 128):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = F.cross_entropy(functional_call(base, det, (P.to_eval(D.xtr[ridx[j:j + 128]].to(dev)),)).float() / T, yb[j:j + 128])
        for (k, _), g in zip(det.items(), torch.autograd.grad(l, list(det.values()))):
            W[k] += g.float().abs()
    mean_w = sum(w.sum().item() for w in W.values()) / NT
    W = {k: w / mean_w for k, w in W.items()}
    print(json.dumps({'weight': 'grad', 'T_corner': T}), flush=True)


B1, B2, EPS = 0.9, 0.999, 1e-8


@torch.no_grad()
def actual_path(x):
    """States theta_k before each official step of the exact run (AdamW, recorded views), no graph."""
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


def grads_at(x, e, create_graph):
    """Batch gradient of the official loss at step e of the target path (recorded view)."""
    (idx, vw), = V['epochs'][e]
    params = {k: t.detach().requires_grad_(True) for k, t in path[e].items()}
    xa = P.augment_views(x[idx.to(dev)], vw)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.bf16):
        ce = F.cross_entropy(functional_call(base, params, (xa,)).float(), y[idx.to(dev)])
    return torch.autograd.grad(ce, list(params.values()), create_graph=create_graph)


R = sum(LRS)


def disp_grad(x):
    """--disp: pixel gradient of mean softplus(-q / tau_d), where q = s * d / R and d = -sum_k lr_k mhat_k / (sqrt(vhat_k) + eps)
    is the AdamW displacement of each weight, from the batch gradients g_k at the target path points. A late step that
    disagrees with s but is small next to the first one costs little, because the momentum keeps the first sign.
    Two passes: plain gradients give dL/dg_k (weight by weight), then one double backward per step."""
    G = [[g.float() for g in grads_at(x.detach(), e, False)] for e in range(5)]
    C = [[None] * len(th0) for _ in range(5)]
    loss, agree = 0., 0
    for j, k in enumerate(th0):
        gl = [G[e][j].requires_grad_(True) for e in range(5)]
        m = v = d = 0.
        for e, lr in enumerate(LRS):
            m = B1 * m + (1 - B1) * gl[e]; v = B2 * v + (1 - B2) * gl[e] * gl[e]
            d = d - lr * (m / (1 - B1 ** (e + 1))) / ((v / (1 - B2 ** (e + 1))).clamp_min(1e-30).sqrt() + EPS)
        q = d * S[k] / R
        lk = F.softplus(-q / a.tau_d).sum() / NT
        for e, c in enumerate(torch.autograd.grad(lk, gl)):
            C[e][j] = c
        loss += lk.item(); agree += (q > 0).sum().item()
    del G
    gx = torch.zeros_like(z)
    for e in range(5):
        gs = grads_at(x, e, True)
        lin = sum((g.float() * c).sum() for g, c in zip(gs, C[e]))
        gx += torch.autograd.grad(lin, z, retain_graph=True)[0]
        del gs, lin
    return gx, loss, agree / NT


def step_loss(x, e, states=None):
    (idx, vw), = V['epochs'][e]
    params = {k: t.detach().requires_grad_(True) for k, t in (states or path)[e].items()}
    xa = P.augment_views(x[idx.to(dev)], vw)
    with sdpa_kernel(SDPBackend.MATH), torch.autocast('cuda', dtype=torch.bfloat16, enabled=a.bf16):
        ce = F.cross_entropy(functional_call(base, params, (xa,)).float(), y[idx.to(dev)])
    gs = torch.autograd.grad(ce, list(params.values()), create_graph=True)
    loss, agree = 0., 0.
    for (k, _), g in zip(params.items(), gs):
        g = g.float()
        r = g.detach().pow(2).mean().sqrt().clamp_min(1e-20)
        m = g * S[k]
        sp = F.softplus(m / (a.tau * r)) if a.loss == 'softplus' else torch.sigmoid(m / (a.tau * r))
        if CW is not None:
            sp = sp * CW[k]
        loss = loss + ((sp * W[k]).sum() if W is not None else sp.sum()) / NT
        agree += ((m < 0) | (S[k] == 0)).sum().item()
    return loss, agree / NT


if a.pfd_w > 0:
    _argv = sys.argv
    sys.argv = ['pfd3.py', '--init', a.init, '--offload', '--fixed-head-seed', '42', '--views', a.views, '--fp32-inner',
                '--outer-T', '0', '--outer-T-mult', str(a.pfd_tmult), '--real-bs', str(a.real_bs)]
    import pfd3 as Q
    sys.argv = _argv
    Q.USE_VIEWS[0] = True
    QHEAD = Q.sample_head()


def pfd_grad(x):
    thK, saved, K = Q.forward(x.detach(), QHEAD)
    G, rl = Q.real_grad(thK, D.sample_train(a.real_bs))
    del thK
    gx = Q.backward(x.detach().clone().requires_grad_(True), saved, G)
    return gx, rl


log = open(f'runs/signmatch_{a.tag}.jsonl', 'w')
best = -1
for it in range(a.iters + 1):
    if it % a.eval_every == 0:
        xq = (pix(z.detach()) * 255).round().div(255).cpu()
        ex = D.train_eval_views(xq, y.cpu(), 42, V, last_only=True)[-1]
        rec = {'it': it, 'exact_acc': round(ex, 4), 'exact_score': round(D.score(ex, len(y)), 4)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
        torch.save({'x01': xq, 'y': y.cpu(), 'it': it}, f'cache/sm_{a.tag}_it{it}.pt')
        if ex > best:
            best = ex
            torch.save({'x01': xq, 'y': y.cpu(), 'it': it}, f'cache/sm_{a.tag}_best.pt')
    if it == a.iters:
        break
    t0 = time.time()
    x = pix(z)
    g = torch.zeros_like(z)
    agrees, losses = [], []
    states = actual_path(x.detach()) if a.path == 'actual' else None
    if a.disp:
        g, l, ag = disp_grad(x)
        agrees.append(round(ag, 4)); losses.append(round(l, 6))
    for e in ([] if a.disp else STEPS):
        l, ag = step_loss(x, e, states)
        w = LRS[e] / LRS[0] if a.wlr else 1.
        g += torch.autograd.grad(l * w, z, retain_graph=True)[0]
        agrees.append(round(ag, 4)); losses.append(round(l.item(), 5))
    rl = None
    if a.pfd_w > 0:
        gp, rl = pfd_grad(x)
        if not ADAM:
            gp = gp * x.detach() * (1 - x.detach())
        g = g / g.pow(2).mean().sqrt().clamp_min(1e-30) + a.pfd_w * gp / gp.pow(2).mean().sqrt().clamp_min(1e-30)
    eps = a.lr_end + (a.lr - a.lr_end) * (1 + math.cos(math.pi * it / a.iters)) / 2
    if a.warmup:
        eps *= min(1., (it + 1) / a.warmup)
    with torch.no_grad():
        if ADAM:
            mo.mul_(0.9).add_(g, alpha=0.1); vo.mul_(0.99).addcmul_(g, g, value=0.01)
            vh = vo / (1 - 0.99 ** (it + 1))
            z -= eps / 255 * (mo / (1 - 0.9 ** (it + 1))) / (vh.sqrt() + 1e-8 * vh.sqrt().mean())
            z.clamp_(0, 1)
        else:
            z -= eps * g / g.abs().max().clamp_min(1e-12)
    if it % 5 == 0:
        rec = {'it': it, 'agree': agrees, 'loss': losses, 'real': rl, 'sec': round(time.time() - t0, 2),
               'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
        print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
