"""How big should one pixel update be? Meta-gradient from `--dir-heads` heads (as in a PFD iteration), then the exact
validation CE / accuracy after the K official steps on `--test` *other* (head, shuffle, crops) setups, for steps
along g (linear), sign(g) and random controls, sized by their mean |dx| in 1/255 units, in float and after uint8 rounding.
Paired: every candidate is scored on the same test setups.

usage: diag_step.py --init cache/pfd3_n120_best.pt
"""
import argparse, json, random, sys
import torch, torch.nn.functional as F
from torch.func import functional_call

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--dir-heads', type=int, default=2)
ap.add_argument('--test', type=int, default=4)
ap.add_argument('--val', type=int, default=2048)
ap.add_argument('--tag', default='')
ap.add_argument('--kinds', default='lin,sign,rand_lin,rand_sign')
args = ap.parse_args()
sys.argv = ['pfd3.py', '--init', args.init, '--offload']
import pfd3 as Q
D, P, dev = Q.D, Q.P, Q.dev

x = D.load_set(args.init)[0].to(dev)
y = Q.y

# direction: same estimator as a PFD iteration, but in pixel space
torch.manual_seed(1000); random.seed(1000)
gx = torch.zeros_like(x)
for _ in range(args.dir_heads):
    thK, saved, K = Q.forward(x, Q.sample_head())
    G, rl = Q.real_grad(thK, torch.randperm(len(D.xtr))[:Q.a.real_bs])
    del thK
    gx += Q.backward(x.clone().requires_grad_(True), saved, G)
    del saved, G

g = torch.Generator().manual_seed(7)
vidx = torch.randperm(len(D.xva), generator=g)[:args.val]
xv, yv = D.xva[vidx], D.yva[vidx].to(dev)
setups = []
for s in range(args.test):
    torch.manual_seed(2000 + s)
    setups.append((Q.sample_head(), 3000 + s))


@torch.no_grad()
def val_metrics(th):
    ce, acc = 0.0, 0.0
    for j in range(0, len(xv), 256):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            o = functional_call(Q.base, th, (P.to_eval(xv[j:j + 256].to(dev)),)).float()
        ce += F.cross_entropy(o, yv[j:j + 256], reduction='sum').item()
        acc += (o.argmax(1) == yv[j:j + 256]).sum().item()
    return ce / len(xv), acc / len(xv)


def score(xc):
    ce, acc = [], []
    for head, seed in setups:
        torch.manual_seed(seed); random.seed(seed)     # same shuffle + crops for every candidate
        th, saved, _ = Q.forward(xc, head)
        del saved
        c, a_ = val_metrics(th)
        ce.append(c); acc.append(a_)
    return sum(ce) / len(ce), sum(acc) / len(acc), ce


Q.a.window = 0      # forward() stores nothing: scoring only
base_ce, base_acc, base_list = score(x)
print(json.dumps({'init': args.init, 'n': len(x), 'K': K, 'base_ce': round(base_ce, 4), 'base_acc': round(base_acc, 4),
                  'base_per_setup': [round(c, 4) for c in base_list]}), flush=True)
rows = []
rnd = torch.randn(x.shape, generator=torch.Generator().manual_seed(11)).to(dev)
GRID = {'lin': (0.03, 0.1, 0.3, 1, 3), 'sign': (1, 3), 'rand_lin': (0.3, 1), 'rand_sign': (1, 3)}


def lowpass(t, f):
    # smooth version of a pixel direction: f x f block means, bilinear upsampling back to 64x64 (f=64: one shift per
    # image and channel)
    lo = F.avg_pool2d(t, f)
    return lo.expand_as(t).contiguous() if f == t.shape[-1] else F.interpolate(lo, size=t.shape[-2:], mode='bilinear')


for f in (4, 8, 16, 64):
    GRID[f'lp{f}'] = (0.3, 1, 3, 10)
for kind in args.kinds.split(','):
    # rand_lin: Gaussian with the meta-gradient's per-pixel magnitudes (|g| ⊙ random sign): same "shape", no signal
    dirn = lowpass(gx, int(kind[2:])) if kind.startswith('lp') else \
        {'lin': gx, 'sign': gx.sign(), 'rand_lin': gx.abs() * rnd.sign(), 'rand_sign': rnd.sign()}[kind]
    unit = dirn / dirn.abs().mean()                    # mean |dx| = 1 (in [0,1] units) before scaling
    for m255 in GRID[kind]:
        xc = (x - unit * m255 / 255).clamp(0, 1)
        for q in ((True,) if kind.startswith('lp') else (False, True)):
            xe = (xc * 255).round() / 255 if q else xc
            ce, acc, lst = score(xe)
            rec = {'kind': kind, 'mean_dx_255': m255, 'quant': q,
                   'changed_px': round(((xe - x).abs() > 0.5 / 255).float().mean().item(), 4),
                   'dCE': round(ce - base_ce, 4), 'dAcc': round(acc - base_acc, 4),
                   'dCE_per_setup': [round(c - b, 4) for c, b in zip(lst, base_list)]}
            print(json.dumps(rec), flush=True); rows.append(rec)
json.dump(rows, open(f'runs/diag_step{args.tag}.json', 'w'))
