"""Does proto.py reproduce the official baseline.py number on the same image set? (official: rand_ipc1_s0 -> 0.2220 at seed 42)"""
import sys, csv, numpy as np, torch, torch.nn.functional as F
from PIL import Image
import proto as P

d = 'sets/rand_ipc1_s0'
rows = list(csv.DictReader(open(f'{d}/train.csv')))
x = torch.from_numpy(np.stack([np.asarray(Image.open(f'{d}/train/{r["filename"]}').convert('RGB')) for r in rows]))
y = torch.tensor([P.CLASSES.index(r['label']) for r in rows])
xva, yva = P.load_split('validation')
for seed in [42, 0, 1]:
    m, opt, sch = P.fresh_model(seed); accs = []
    xs = x.cuda().permute(0, 3, 1, 2).float().div(255)
    for ep in range(5):
        opt.zero_grad(set_to_none=True)
        F.cross_entropy(m(P.augment(xs)), y.cuda()).backward(); opt.step(); sch.step()
        accs.append(round(P.evaluate(m, xva, yva), 4))
    print(f'proto seed={seed} per-epoch={accs}', flush=True)
