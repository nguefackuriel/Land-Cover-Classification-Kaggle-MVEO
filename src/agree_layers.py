"""Per-tensor sign agreement between the exact run's per-step updates and a target corner (sign(theta*-theta0)).
usage: PROTO_PIL=1 agree_layers.py <set.pt> <views.pt> <target.pt>"""
import sys, re, collections, torch
src, views, target = sys.argv[1:4]
sys.argv = ['signmatch.py', '--init', src, '--views', views, '--target', target]
exec(open('src/signmatch.py').read().split("log = open(")[0])
x = torch.load(src)['x01'].to(dev)
states = actual_path(x)
th5 = actual_path.__globals__  # noqa
# final state: replay one more step via actual_path internals is awkward; use per-step gradients instead
groups = collections.defaultdict(lambda: [0, 0, 0])
for e in range(5):
    (idx, vw), = V['epochs'][e]
    params = {k: t.detach().requires_grad_(True) for k, t in states[e].items()}
    ce = F.cross_entropy(functional_call(base, params, (P.augment_views(x[idx.to(dev)], vw),)).float(), y[idx.to(dev)])
    gs = torch.autograd.grad(ce, list(params.values()))
    for (k, _), g in zip(params.items(), gs):
        kk = re.sub(r'encoder_layer_(\d+)', lambda mm: 'L' + ('0-3' if int(mm.group(1)) < 4 else '4-7' if int(mm.group(1)) < 8 else '8-11'), k)
        kk = kk.replace('encoder.layers.', '')
        ag = ((-g.sign()) == S[k]).float().sum().item()
        groups[(e, kk)][0] += ag; groups[(e, kk)][1] += g.numel()
for e in (0, 2, 4):
    print(f'--- step {e + 1}')
    rows = sorted([(k, v[0] / v[1], v[1]) for (ee, k), v in groups.items() if ee == e], key=lambda r: -r[2])
    for k, a_, nn in rows[:24]:
        print(f'{k:55s} {a_:.3f} {nn:>9d}')
