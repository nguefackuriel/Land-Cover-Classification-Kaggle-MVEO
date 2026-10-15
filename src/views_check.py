"""Does the replica with the recorded official views (PROTO_PIL=1, fp32 train and eval, seed head) reproduce the
official per-epoch validation accuracy? usage: PROTO_PIL=1 views_check.py cache/set.pt cache/views.pt"""
import json, sys, torch
import dd_lib as D, proto as P

P.evaluate = (lambda f: (lambda m, xu8, y, bs=256: f32_eval(m, xu8, y, bs)))(None)


@torch.no_grad()
def f32_eval(m, xu8, y, bs=256):
    m.eval(); correct = 0
    for i in range(0, len(xu8), bs):
        correct += (m(P.to_eval(xu8[i:i + bs].cuda())).argmax(1).cpu() == y[i:i + bs]).sum().item()
    m.train(); return correct / len(xu8)


(x, y), views = D.load_set(sys.argv[1]), torch.load(sys.argv[2])
accs = D.train_eval_views(x, y, views['seed'], views)
print(json.dumps({'set': sys.argv[1], 'seed': views['seed'], 'pil': P.PIL_ROUND, 'per_epoch': [round(a, 4) for a in accs]}))
