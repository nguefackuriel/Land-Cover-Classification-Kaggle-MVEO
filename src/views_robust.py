"""How fragile is a seed-targeted set to small numerical differences (another GPU / library at verification)?
Replays the exact official views (PROTO_PIL=1, fp32 eval) under several numeric settings and reports the final
validation accuracy. usage: PROTO_PIL=1 views_robust.py cache/views.pt cache/a.pt [cache/b.pt ...]"""
import json, sys, torch
import dd_lib as D, proto as P


@torch.no_grad()
def f32_eval(m, xu8, y, bs=256):
    m.eval(); correct = 0
    for i in range(0, len(xu8), bs):
        correct += (m(P.to_eval(xu8[i:i + bs].cuda())).argmax(1).cpu() == y[i:i + bs]).sum().item()
    m.train(); return correct / len(xu8)


P.evaluate = f32_eval
SETTINGS = {'official': (False, True), 'tf32_matmul': (True, True), 'no_tf32_conv': (False, False),
            'tf32mm_fp32conv': (True, False)}
views = torch.load(sys.argv[1])
for f in sys.argv[2:]:
    x, y = D.load_set(f); res = {}
    for name, (mm, conv) in SETTINGS.items():
        torch.backends.cuda.matmul.allow_tf32 = mm; torch.backends.cudnn.allow_tf32 = conv
        res[name] = round(D.train_eval_views(x, y, views['seed'], views)[-1], 4)
    torch.backends.cuda.matmul.allow_tf32 = False; torch.backends.cudnn.allow_tf32 = True
    print(json.dumps({'set': f, **res}), flush=True)
