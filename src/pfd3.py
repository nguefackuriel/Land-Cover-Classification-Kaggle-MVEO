"""PFD3 — pfd2.py with a per-step checkpointed reverse pass: memory O(one inner step) instead of O(window).

Forward: the exact official schedule (shuffle each epoch, batches of 32 with the leftover kept, AdamW 3e-5 / wd .05,
per-epoch cosine lr) without any graph, storing (theta, m, v) before each of the first `--window` steps together with
that step's batch and augmentation RNG state. Backward: for t = window..1, rebuild step t's graph from its stored
state (straight-through Adam, as in pfd.py) and pull the adjoints of (theta, m, v) back through it, collecting
d/dx on the way. Steps after the window are identity in the backward pass, as in pfd2.py. Same meta-gradient as
pfd2.py's full-graph window (see --check), so the window can now cover the whole trajectory.

usage: pfd3.py --init cache/medoids_ipc12.pt --window 20 --iters 300 --tag n120
       pfd3.py --init cache/pfd2_w6_n40_best.pt --window 4 --check     # compare with the full-graph gradient
"""
import argparse, contextlib, json, math, random, sys, time
import torch, torch.nn.functional as F
from torch.func import functional_call
from torch.nn.attention import SDPBackend, sdpa_kernel

import dd_lib as D
import proto as P

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--window', type=int, default=100)         # >= K: full unroll
ap.add_argument('--iters', type=int, default=300)
ap.add_argument('--opt', default='maxnorm', choices=['maxnorm', 'adam'])  # adam: plain pixels, lr in 1/255 levels
ap.add_argument('--pixel-noise', type=float, default=0.)   # randomized smoothing: N(0, s^2) in 1/255 levels per sample
ap.add_argument('--lr', type=float, default=0.05)
ap.add_argument('--lr-end', type=float, default=0.01)
ap.add_argument('--tau', type=float, default=0.3)
ap.add_argument('--heads', type=int, default=2)
ap.add_argument('--real-bs', type=int, default=512)
ap.add_argument('--eval-every', type=int, default=25)
ap.add_argument('--offload', action='store_true')          # keep the stored states in pinned host memory
ap.add_argument('--check', action='store_true')
ap.add_argument('--fp32-inner', action='store_true')
ap.add_argument('--tf32', action='store_true')
ap.add_argument('--save-every-eval', action='store_true')
ap.add_argument('--views', default='')                     # record_views.py file: exact official batches + crops/flips
ap.add_argument('--views-prob', type=float, default=1.)    # share of meta-gradient samples using them (rest: random)
ap.add_argument('--select-exact', action='store_true')
ap.add_argument('--line-search', default='')    # e.g. "0,0.5,1,2,4": try these multiples of the step, keep the best
ap.add_argument('--ls-real', type=int, default=2048)   # fixed training images that score the line-search candidates      # best checkpoint = exact-view accuracy, not the 3-seed mean    # also keep cache/pfd3_<tag>_it<it>.pt at each eval
ap.add_argument('--fixed-head-seed', type=int, default=-1)  # >= 0: always the head baseline.py draws for this seed
                                                            # (analysis of seed-specific overfitting, not for submissions)               # with --fp32-inner: TF32 matmuls (10-bit mantissa vs 7 for bf16)
ap.add_argument('--tag', default='check')
ap.add_argument('--no-seed-eval', action='store_true')   # exact runs: skip the 3 random-seed evals, keep only the exact one
ap.add_argument('--target', default='')        # state dict of a good reachable network (ceiling3.py): pull theta_K to it
ap.add_argument('--target-mode', default='l2', choices=['l2', 'lin'])   # l2: |theta_K - t|^2 ; lin: -<t - theta_0, theta_K>
ap.add_argument('--target-w', type=float, default=-1.)  # <0: target only; >=0: real grad + w * (target grad scaled to its norm)
ap.add_argument('--outer-T', type=float, default=1.)       # outer CE on logits / T; 0: T fitted on each real batch
                                                           # (scale-free: no credit for merely sharper logits)
a = ap.parse_args()

dev = 'cuda'
if a.tf32:
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
LRS = [3e-5 * (1 + math.cos(math.pi * e / 5)) / 2 for e in range(5)]
B1, B2, EPS, WD = 0.9, 0.999, 1e-8, 0.05
HEAD_BOUND = 1 / math.sqrt(768)

base, _, _ = P.fresh_model(0)
base.eval()
for p in base.parameters():
    p.requires_grad_(False)
theta0 = {k: p.detach() for k, p in base.named_parameters()}

x0, y = D.load_set(a.init)     # submissions/<set> folder, pfd3 .pt, or inits/medoids_*.pt
y = y.to(dev)
n = len(y)
X_SPACE = a.opt == 'adam'      # adam: pixels in [0,1] directly (diag_step.py: +-1 level sign steps work best);
                               # maxnorm: logits, z -= eps * g / max|g|
z = (x0.clone() if X_SPACE else torch.logit(x0.clamp(0.02, 0.98))).to(dev).requires_grad_(True)
mo, vo = torch.zeros_like(z), torch.zeros_like(z)     # outer Adam state (--opt adam)
B1O, B2O = 0.9, 0.99


def pixels(t):
    return t.clamp(0, 1) if X_SPACE else torch.sigmoid(t)


def adam_dir(m, v, g, t):
    m = B1 * m + (1 - B1) * g
    v = B2 * v + (1 - B2) * g * g
    mh, vh = m / (1 - B1 ** t), v / (1 - B2 ** t)
    exact = mh / (vh.sqrt() + EPS)
    smooth = mh / (vh + a.tau ** 2 * vh.detach().mean() + EPS ** 2).sqrt()
    return m, v, smooth + (exact - smooth).detach()


def inner_loss(params, xb, yb):
    with torch.autocast('cuda', dtype=torch.bfloat16, enabled=not a.fp32_inner):
        return F.cross_entropy(functional_call(base, params, (xb,)).float(), yb)


_fixed = {}
VIEWS = torch.load(a.views) if a.views else None
USE_VIEWS = [False]


def sample_head():
    if a.fixed_head_seed >= 0:
        if not _fixed:
            m, _, _ = P.fresh_model(a.fixed_head_seed)
            _fixed['h'] = (m.heads.head.weight.detach().clone(), m.heads.head.bias.detach().clone())
            del m
        return _fixed['h']
    return (torch.empty(10, 768, device=dev).uniform_(-HEAD_BOUND, HEAD_BOUND),
            torch.empty(10, device=dev).uniform_(-HEAD_BOUND, HEAD_BOUND))


def init_state(head):
    th = dict(theta0)
    th['heads.head.weight'], th['heads.head.bias'] = head
    return th, {k: torch.zeros_like(t) for k, t in th.items()}, {k: torch.zeros_like(t) for k, t in th.items()}


_pin = {}          # pinned host buffers, allocated once per (window slot, tensor) and reused every iteration


def stash(d, slot):
    if not a.offload:
        return d
    out = {}
    for k, t in d.items():
        buf = _pin.get((slot, k))
        if buf is None:
            buf = _pin[(slot, k)] = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        out[k] = buf.copy_(t, non_blocking=True)   # same stream as the reads in unstash: ordered
    return out


def unstash(d):
    return {k: t.to(dev, non_blocking=True) for k, t in d.items()} if a.offload else d


def forward(x01, head):
    """Exact official trajectory without graph. Returns theta_K and the stored states of the window steps."""
    th, m, v = init_state(head)
    saved, t = [], 0
    for e, lr in enumerate(LRS):
        if USE_VIEWS[0]:
            batches = [(idx.to(dev), vw) for idx, vw in VIEWS['epochs'][e]]
        else:
            perm = torch.randperm(n, device=dev)
            batches = [(perm[j:j + 32], None) for j in range(0, n, 32)]
        for b, vw in batches:
            t += 1
            if t <= a.window:
                saved.append((stash(th, (t, 0)), stash(m, (t, 1)), stash(v, (t, 2)), b,
                              vw if vw is not None else random.getstate(), lr, t))
            params = {k: p.detach().requires_grad_(True) for k, p in th.items()}
            with torch.no_grad():
                xb = P.augment_views(x01[b], vw) if vw is not None else P.augment(x01[b])
            with sdpa_kernel(SDPBackend.MATH) if a.check else contextlib.nullcontext():   # --check: as full
                gs = torch.autograd.grad(inner_loss(params, xb, y[b]), list(params.values()))
            with torch.no_grad():       # new dicts every step: the stored references must stay valid
                nth, nm, nv = {}, {}, {}
                for (k, p), g in zip(params.items(), gs):
                    if a.check:         # same rounding as the full graph: the trajectory is chaotic (cos .987 else)
                        nm[k], nv[k], u = adam_dir(m[k], v[k], g, t)
                        nth[k] = p * (1 - lr * WD) - lr * u
                        continue
                    nm[k] = B1 * m[k] + (1 - B1) * g
                    nv[k] = B2 * v[k] + (1 - B2) * g * g
                    mh, vh = nm[k] / (1 - B1 ** t), nv[k] / (1 - B2 ** t)
                    nth[k] = p * (1 - lr * WD) - lr * mh / (vh.sqrt() + EPS)
            th, m, v = nth, nm, nv
    return th, saved, t


def fit_T(th, ridx):
    """Temperature minimising the real CE at theta_K (envelope theorem: T is treated as a constant in the gradient)."""
    with torch.no_grad(), torch.autocast('cuda', dtype=torch.bfloat16):
        o = torch.cat([functional_call(base, th, (P.to_eval(D.xtr[ridx[j:j + 128]].to(dev)),)).float()
                       for j in range(0, len(ridx), 128)])
    yb = D.ytr[ridx].to(dev)
    Ts = torch.logspace(-2, 1, 121, device=dev)
    ce = torch.stack([F.cross_entropy(o / T, yb) for T in Ts])
    return Ts[ce.argmin()].item()


@torch.no_grad()
def scalefree_ce(th, ridx):
    """Outer loss value (no gradient) on fixed real images: CE at the best temperature (or at --outer-T)."""
    with torch.autocast('cuda', dtype=torch.bfloat16):
        o = torch.cat([functional_call(base, th, (P.to_eval(D.xtr[ridx[j:j + 256]].to(dev)),)).float()
                       for j in range(0, len(ridx), 256)])
    yb = D.ytr[ridx].to(dev)
    Ts = torch.logspace(-2, 1, 121, device=dev) if a.outer_T <= 0 else torch.tensor([a.outer_T], device=dev)
    return min(F.cross_entropy(o / T, yb).item() for T in Ts)


def ls_score(zc, head):
    """Line-search objective of candidate pixels: exact forward run (no stored states), then the outer loss."""
    w = a.window; a.window = 0
    USE_VIEWS[0] = bool(VIEWS)
    with torch.enable_grad():          # the inner training needs gradients even inside the no_grad update block
        th, _, _ = forward(pixels(zc.detach()), head)
    a.window = w
    return scalefree_ce(th, LS_IDX)


def real_grad(th, ridx):
    det = {k: t.detach().requires_grad_(True) for k, t in th.items()}
    G = {k: torch.zeros_like(t) for k, t in det.items()}
    total = 0.0
    T = a.outer_T if a.outer_T > 0 else fit_T(th, ridx)
    real_grad.last_T = T
    for j in range(0, len(ridx), 128):
        xb = P.to_eval(D.xtr[ridx[j:j + 128]].to(dev))
        with torch.autocast('cuda', dtype=torch.bfloat16):
            l = F.cross_entropy(functional_call(base, det, (xb,)) / T, D.ytr[ridx[j:j + 128]].to(dev)) * len(xb) / len(ridx)
        for (k, _), g in zip(det.items(), torch.autograd.grad(l, list(det.values()))):
            G[k] += g.float()
        total += l.item()
    return G, total


def backward(x_leaf, saved, G):
    """Reverse pass over the stored window steps; returns d CE_real / d x01."""
    lam_th, lam_m, lam_v = G, None, None
    xg = torch.zeros_like(x_leaf)
    for th, m, v, b, rs, lr, t in reversed(saved):
        th, m, v = unstash(th), unstash(m), unstash(v)
        th_in = {k: p.detach().requires_grad_(True) for k, p in th.items()}
        mv = t > 1                       # m_0 = v_0 = 0 are constants
        m_in = {k: q.detach().requires_grad_(mv) for k, q in m.items()}
        v_in = {k: q.detach().requires_grad_(mv) for k, q in v.items()}
        if isinstance(rs, torch.Tensor):        # exact recorded views
            xa = P.augment_views(x_leaf[b], rs)
        else:
            random.setstate(rs)
            xa = P.augment(x_leaf[b])
        with sdpa_kernel(SDPBackend.MATH):
            gs = torch.autograd.grad(inner_loss(th_in, xa, y[b]), list(th_in.values()), create_graph=True)
        outs, gouts = [], []
        for (k, p), g in zip(th_in.items(), gs):
            nm, nv, u = adam_dir(m_in[k], v_in[k], g, t)
            outs.append(p * (1 - lr * WD) - lr * u); gouts.append(lam_th[k])
            if lam_m is not None:
                outs += [nm, nv]; gouts += [lam_m[k], lam_v[k]]
        del gs
        keys = list(th_in)
        inputs = list(th_in.values()) + (list(m_in.values()) + list(v_in.values()) if mv else []) + [x_leaf]
        grads = torch.autograd.grad(outs, inputs, gouts, allow_unused=True)
        del outs, gouts
        zero = lambda i, ref: grads[i] if grads[i] is not None else torch.zeros_like(ref)
        P_ = len(keys)
        lam_th = {k: zero(i, th_in[k]) for i, k in enumerate(keys)}
        if mv:
            lam_m = {k: zero(P_ + i, m_in[k]) for i, k in enumerate(keys)}
            lam_v = {k: zero(2 * P_ + i, v_in[k]) for i, k in enumerate(keys)}
        if grads[-1] is not None:
            xg += grads[-1]
        del grads
    return xg


TGT = None
if a.target:
    _sd = torch.load(a.target)
    TGT = {k: _sd[k].to(dev).float() for k in theta0}
    TGT['heads.head.weight'], TGT['heads.head.bias'] = _sd['heads.head.weight'].to(dev), _sd['heads.head.bias'].to(dev)


def target_grad(th, G):
    """Outer gradient wrt theta_K that pulls the trajectory end point toward the target network."""
    if a.target_mode == 'l2':
        Gt = {k: (th[k].float() - TGT[k]) for k in th}
    else:
        h0 = sample_head()
        ref = dict(theta0); ref['heads.head.weight'], ref['heads.head.bias'] = h0
        Gt = {k: -(TGT[k] - ref[k].float()) for k in th}
    if a.target_w < 0:
        return Gt
    nr = math.sqrt(sum((g.float() ** 2).sum().item() for g in G.values()))
    nt = math.sqrt(sum((g ** 2).sum().item() for g in Gt.values()))
    return {k: G[k] + a.target_w * nr / max(nt, 1e-30) * Gt[k] for k in G}


def meta_grad(head, ridx):
    """d CE_real(theta_K) / d z for one head (checkpointed). Returns (grad, real loss, K)."""
    USE_VIEWS[0] = bool(VIEWS) and random.random() < a.views_prob
    x01 = pixels(z.detach())
    xs = (x01 + a.pixel_noise / 255 * torch.randn_like(x01)).clamp(0, 1) if a.pixel_noise else x01
    thK, saved, K = forward(xs, head)
    G, rl = real_grad(thK, ridx)
    if TGT is not None:
        G = target_grad(thK, G)
        meta_grad.tdist = math.sqrt(sum(((thK[k].float() - TGT[k]) ** 2).sum().item() for k in thK))
    del thK
    x_leaf = xs.clone().requires_grad_(True)       # grad at x + delta = grad of E_delta L(x + delta) wrt x
    xg = backward(x_leaf, saved, G)
    return (xg if X_SPACE else xg * x01 * (1 - x01)), rl, K


def meta_grad_fullgraph(head, ridx):
    """pfd2.py's full-graph window gradient, for --check only."""
    zz = z.detach().clone().requires_grad_(True)
    x01 = pixels(zz)
    th, m, v = init_state(head)
    t, th_window = 0, None
    for lr in LRS:
        perm = torch.randperm(n, device=dev)
        for j in range(0, n, 32):
            b = perm[j:j + 32]
            t += 1
            if t <= a.window:
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
    if th_window is None:
        th_window, th = th, {k: tt.detach() for k, tt in th.items()}
    G, rl = real_grad(th, ridx)
    keys = [k for k in th_window if th_window[k].requires_grad]
    torch.autograd.backward([th_window[k] for k in keys], [G[k] for k in keys])
    return zz.grad.detach(), rl


if __name__ == '__main__':      # importable: diag scripts reuse forward/real_grad/backward
    if a.check:
        head = sample_head()
        ridx = torch.randperm(len(D.xtr))[:a.real_bs]
        out = {}
        for name, fn in (('ckpt', meta_grad), ('ckpt2', meta_grad), ('full', meta_grad_fullgraph), ('full2', meta_grad_fullgraph)):
            torch.manual_seed(123); random.seed(123)
            torch.cuda.reset_peak_memory_stats(); t0 = time.time()
            r = fn(head, ridx)
            out[name] = r[0]
            print(name, f'real_loss={r[1]:.5f} sec={time.time() - t0:.1f} '
                  f'peak_gb={torch.cuda.max_memory_allocated() / 2**30:.1f}', flush=True)
        for p, q in (('ckpt', 'full'), ('ckpt', 'ckpt2'), ('full', 'full2'), ('ckpt2', 'full2')):
            c, f = out[p].flatten(), out[q].flatten()
            print(f'n={n} window={a.window} {p} vs {q}: cos={F.cosine_similarity(c, f, dim=0).item():.5f}  '
                  f'norm ratio={(c.norm() / f.norm()).item():.4f}  rel_err={((c - f).norm() / f.norm()).item():.4f}')
        sys.exit()

    LS_MULT = [float(v) for v in a.line_search.split(',')] if a.line_search else []
    LS_IDX = torch.randperm(len(D.xtr), generator=torch.Generator().manual_seed(7))[:a.ls_real]
    ls_log = []
    log = open(f'runs/pfd3_{a.tag}.jsonl', 'w')
    best = -1
    for it in range(a.iters + 1):
        if it % a.eval_every == 0:
            xq = (pixels(z.detach()) * 255).round().div(255).cpu()
            accs = [] if a.no_seed_eval else [round(D.train_eval(xq, y.cpu(), s), 4) for s in (42, 0, 1)]
            mean = sum(accs) / 3 if accs else 0.
            rec = {'it': it, 'n': n, 'val_accs': accs, 'mean': round(mean, 4), 'score_mean': round(D.score(mean, n), 4)}
            if VIEWS:
                ex = D.train_eval_views(xq, y.cpu(), VIEWS['seed'], VIEWS, last_only=True)[-1]
                rec.update({'exact_seed': VIEWS['seed'], 'exact_acc': round(ex, 4), 'exact_score': round(D.score(ex, n), 4)})
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
            if a.save_every_eval:
                torch.save({'x01': xq, 'y': y.cpu(), 'it': it, 'accs': accs},
                           f'cache/pfd3_{a.tag}_it{it}.pt')
            crit = rec['exact_acc'] if (VIEWS and a.select_exact) else mean
            if crit > best:
                best = crit
                torch.save({'x01': xq, 'y': y.cpu(), 'it': it, 'accs': accs},
                           f'cache/pfd3_{a.tag}_best.pt')
        if it == a.iters:
            break
        t0 = time.time()
        g, real_loss = torch.zeros_like(z), 0.0
        for _ in range(a.heads):
            gh, rl, K = meta_grad(sample_head(), torch.randperm(len(D.xtr))[:a.real_bs])
            g += gh; real_loss += rl / a.heads
        if torch.isfinite(g).all():
            eps = a.lr_end + (a.lr - a.lr_end) * (1 + math.cos(math.pi * it / a.iters)) / 2
            with torch.no_grad():
                if X_SPACE:
                    mo.mul_(B1O).add_(g, alpha=1 - B1O); vo.mul_(B2O).addcmul_(g, g, value=1 - B2O)
                    vh = vo / (1 - B2O ** (it + 1))
                    z -= eps / 255 * (mo / (1 - B1O ** (it + 1))) / (vh.sqrt() + 1e-8 * vh.sqrt().mean())
                    z.clamp_(0, 1)
                elif a.line_search:
                    d = g / g.abs().max().clamp_min(1e-12)
                    head = sample_head()
                    cands = [(ls_score(z - mlt * eps * d, head), mlt) for mlt in LS_MULT]
                    best_l, best_m = min(cands)
                    z -= best_m * eps * d
                    ls_log.append(best_m)
                else:
                    z -= eps * g / g.abs().max().clamp_min(1e-12)
        else:
            print(json.dumps({'it': it, 'skipped': 'non-finite grad'}), flush=True)
        if it % 5 == 0:
            rec = {'it': it, 'K': K, 'window': min(a.window, K), 'real_loss_after_K': round(real_loss, 4),
                   'tdist': round(getattr(meta_grad, 'tdist', 0.), 6),
                   'T': round(getattr(real_grad, 'last_T', 1.), 4), 'ls_mult': ls_log[-5:],
                   'sec': round(time.time() - t0, 2), 'peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 1)}
            print(json.dumps(rec), flush=True); log.write(json.dumps(rec) + '\n'); log.flush()
