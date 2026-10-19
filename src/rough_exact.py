"""How rough is the exact seed-42 objective around a set? Exact meta-gradient (recorded views, seed-42 head, fixed 4096
train images) vs random directions, at several step sizes (mean |dx| in 1/255 levels), scored by the exact replay:
calibrated train CE + train acc on the fixed 4096 images and val acc (4050). Float and uint8-rounded pixels.
usage: PROTO_PIL=1 rough_exact.py <set.pt> <views.pt>"""
import json, sys, torch, torch.nn.functional as F
from torch.func import functional_call
src, views = sys.argv[1], sys.argv[2]
sys.argv = ['pfd3.py', '--init', src, '--offload', '--fixed-head-seed', '42', '--views', views, '--fp32-inner', '--outer-T', '0']
import pfd3 as Q
D, P, dev = Q.D, Q.P, Q.dev
Q.USE_VIEWS[0] = True
head = Q.sample_head()
x0 = Q.pixels(Q.z.detach()).clone()
x0 = (x0 * 255).round() / 255
RIDX = torch.randperm(len(D.xtr), generator=torch.Generator().manual_seed(11))[:4096]

@torch.no_grad()
def logits(th, xu8):
    with torch.autocast('cuda', dtype=torch.bfloat16):
        return torch.cat([functional_call(Q.base, th, (P.to_eval(xu8[j:j + 256].to(dev)),)).float() for j in range(0, len(xu8), 256)])

def score(x):
    w = Q.a.window; Q.a.window = 0
    th, _, _ = Q.forward(x, head)
    Q.a.window = w
    o = logits(th, D.xtr[RIDX]); yb = D.ytr[RIDX].to(dev)
    Ts = torch.logspace(-2, 1, 61, device=dev)
    ce = min(F.cross_entropy(o / T, yb).item() for T in Ts)
    tr = (o.argmax(1) == yb).float().mean().item()
    va = (logits(th, D.xva).argmax(1).cpu() == D.yva).float().mean().item()
    return round(ce, 4), round(tr, 4), round(va, 4)

base = score(x0)
print(json.dumps({'kind': 'base', 'ce': base[0], 'train_acc': base[1], 'val': base[2]}), flush=True)
thK, saved, K = Q.forward(x0, head)
G, rl = Q.real_grad(thK, RIDX[:2048])
del thK
gx = Q.backward(x0.clone().requires_grad_(True), saved, G)
del saved, G
gen = torch.Generator(device=dev).manual_seed(3)
for s in (0.1, 0.3, 1.0, 3.0):
    dirs = {'lin': -gx / gx.abs().mean() * s / 255, 'sign': -gx.sign() * s / 255}
    for r in range(2):
        rs = torch.randint(0, 2, x0.shape, generator=gen, device=dev).float() * 2 - 1
        dirs[f'rand{r}'] = rs * s / 255
    for k, d in dirs.items():
        xf = (x0 + d).clamp(0, 1)
        out = {'kind': k, 'size': s}
        out['float'] = score(xf)
        if s >= 0.5:
            out['u8'] = score((xf * 255).round() / 255)
        print(json.dumps(out), flush=True)
