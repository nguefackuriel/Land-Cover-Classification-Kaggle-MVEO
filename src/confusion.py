"""Per-class validation accuracy and top confusions of the proto recipe trained on a set (seeds averaged).
usage: confusion.py cache/pfd3_n120_best.pt [cache/medoids_ipc12.pt ...]"""
import sys, torch
import dd_lib as D, proto as P
import torch.nn.functional as F
NAMES = ['AnnualCrop', 'Forest', 'Herbaceous', 'Highway', 'Industrial', 'Pasture', 'PermCrop', 'Residential', 'River', 'SeaLake']


def train_final(x01, y, seed):
    m, opt, sch = P.fresh_model(seed); x01, y = x01.cuda(), y.cuda()
    for ep in range(5):
        perm = torch.randperm(len(y))
        for j in range(0, len(y), 32):
            b = perm[j:j + 32]; opt.zero_grad(set_to_none=True)
            F.cross_entropy(m(P.augment(x01[b])), y[b]).backward(); opt.step()
        sch.step()
    m.eval(); preds = []
    with torch.no_grad():
        for i in range(0, len(D.xva), 256):
            with torch.autocast('cuda', dtype=torch.bfloat16):
                preds.append(m(P.to_eval(D.xva[i:i + 256].cuda())).argmax(1).cpu())
    return torch.cat(preds)


for f in sys.argv[1:]:
    x, y = D.load_set(f)
    C = torch.zeros(10, 10)
    for s in (42, 0, 1):
        p = train_final(x, y, s)
        C += torch.bincount(D.yva * 10 + p, minlength=100).view(10, 10).float()
    C /= 3
    acc = C.diag() / C.sum(1)
    print(f'{f}  n={len(y)}  overall={C.diag().sum() / C.sum():.4f}')
    print('  per class: ' + '  '.join(f'{NAMES[c]}={acc[c]*100:.1f}' for c in range(10)))
    off = C.clone(); off.fill_diagonal_(0)
    top = off.flatten().argsort(descending=True)[:8]
    print('  top confusions (true->pred, avg count of 4050): ' +
          ', '.join(f'{NAMES[i // 10]}->{NAMES[i % 10]} {off.flatten()[i]:.0f}' for i in top.tolist()))
