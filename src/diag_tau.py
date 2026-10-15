"""Does the straight-through smoothing tau bias the meta-gradient? Same heads / shuffles / crops / real batches for
every tau; each direction is scored like diag_step.py (exact val CE / acc after K official steps on `--test` other
setups, quantized steps along g and sign(g)).

usage: diag_tau.py --init cache/pfd3_n120_best.pt --taus 0.03,0.1,0.3,1,3
"""
import argparse, json, random, sys
import torch, torch.nn.functional as F
from torch.func import functional_call

ap = argparse.ArgumentParser()
ap.add_argument('--init', required=True)
ap.add_argument('--taus', default='0.03,0.1,0.3,1,3')
ap.add_argument('--dir-heads', type=int, default=4)
ap.add_argument('--test', type=int, default=4)
ap.add_argument('--val', type=int, default=2048)
ap.add_argument('--tag', default='')
args = ap.parse_args()
sys.argv = ['pfd3.py', '--init', args.init, '--offload']
import pfd3 as Q
D, P, dev = Q.D, Q.P, Q.dev

x = D.load_set(args.init)[0].to(dev)

torch.manual_seed(1000)
dir_heads = [Q.sample_head() for _ in range(args.dir_heads)]
ridx = [torch.randperm(len(D.xtr))[:Q.a.real_bs] for _ in range(args.dir_heads)]


def direction(tau):
    Q.a.tau, Q.a.window = tau, 100
    gx = torch.zeros_like(x)
    for h, (head, r) in enumerate(zip(dir_heads, ridx)):
        torch.manual_seed(1100 + h); random.seed(1100 + h)        # same shuffle + crops for every tau
        thK, saved, K = Q.forward(x, head)
        G, _ = Q.real_grad(thK, r)
        del thK
        gx += Q.backward(x.clone().requires_grad_(True), saved, G)
        del saved, G
    return gx


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
    Q.a.window = 0
    ce, acc = [], []
    for head, seed in setups:
        torch.manual_seed(seed); random.seed(seed)
        th, _, _ = Q.forward(xc, head)
        c, a_ = val_metrics(th)
        ce.append(c); acc.append(a_)
    return sum(ce) / len(ce), sum(acc) / len(acc)


base_ce, base_acc = score(x)
print(json.dumps({'init': args.init, 'base_ce': round(base_ce, 4), 'base_acc': round(base_acc, 4)}), flush=True)
dirs, rows = {}, []
for tau in map(float, args.taus.split(',')):
    dirs[tau] = direction(tau)
    for kind, m255 in (('lin', 0.03), ('lin', 0.1), ('lin', 0.3), ('sign', 1)):
        dn = dirs[tau] if kind == 'lin' else dirs[tau].sign()
        xe = ((x - dn / dn.abs().mean() * m255 / 255).clamp(0, 1) * 255).round() / 255
        ce, acc = score(xe)
        rec = {'tau': tau, 'kind': kind, 'mean_dx_255': m255, 'dCE': round(ce - base_ce, 4), 'dAcc': round(acc - base_acc, 4)}
        print(json.dumps(rec), flush=True); rows.append(rec)
taus = list(dirs)
cos = {f'{a}|{b}': round(F.cosine_similarity(dirs[a].flatten(), dirs[b].flatten(), dim=0).item(), 3)
       for i, a in enumerate(taus) for b in taus[i + 1:]}
print(json.dumps({'cos_between_taus': cos}), flush=True)
json.dump({'rows': rows, 'cos': cos}, open(f'runs/diag_tau{args.tag}.json', 'w'))
