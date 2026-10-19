"""Exact replay of the recorded official run (PROTO_PIL=1, fp32) for a set: per-class validation accuracy, confusion
counts, and accuracy on 4050 random training images (eval transform). usage: PROTO_PIL=1 views_confusion.py <set> <views.pt>"""
import json, sys, torch
import dd_lib as D, proto as P
import torch.nn.functional as F

NAMES = ['AnnualCrop', 'Forest', 'HerbVeg', 'Highway', 'Industrial', 'Pasture', 'PermCrop', 'Residential', 'River', 'SeaLake']
x, y = D.load_set(sys.argv[1]); views = torch.load(sys.argv[2])
m, opt, sch = P.fresh_model(views['seed']); x, y = x.cuda(), y.cuda()
for batches in views['epochs']:
    for idx, vw in batches:
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(P.augment_views(x[idx.cuda()], vw)), y[idx.cuda()]).backward(); opt.step()
    sch.step()
m.eval()


@torch.no_grad()
def predict(xu8):
    return torch.cat([m(P.to_eval(xu8[i:i + 256].cuda())).argmax(1).cpu() for i in range(0, len(xu8), 256)])


pv = predict(D.xva)
C = torch.bincount(D.yva * 10 + pv, minlength=100).view(10, 10)
g = torch.Generator().manual_seed(123)
tidx = torch.randperm(len(D.xtr), generator=g)[:len(D.xva)]
pt = predict(D.xtr[tidx])
print(json.dumps({'set': sys.argv[1], 'val_acc': round((pv == D.yva).float().mean().item(), 4),
                  'train_subset_acc': round((pt == D.ytr[tidx]).float().mean().item(), 4),
                  'per_class': {NAMES[c]: round(C[c, c].item() / C[c].sum().item(), 3) for c in range(10)},
                  'class_share': {NAMES[c]: round(C[c].sum().item() / C.sum().item(), 3) for c in range(10)}}))
off = C.clone(); off.fill_diagonal_(0)
top = off.flatten().argsort(descending=True)[:10]
print('top confusions (true->pred, count of 4050):', ', '.join(f'{NAMES[i // 10]}->{NAMES[i % 10]} {off.flatten()[i].item()}' for i in top.tolist()))
