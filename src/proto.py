"""Fast GPU re-implementation of the official evaluation protocol (baseline.py) for research loops.
Mirrors: seeding -> build_model head init, AdamW(3e-5, wd .05), per-epoch cosine (T_max=5), batch 32,
RandomResizedCrop(224, .75-1)+H/V flips train aug, Resize256/CenterCrop224 eval. Final numbers must
still come from the unmodified baseline.py."""
import csv, json, os, random, sys
from pathlib import Path
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, 'ext/challenge_2026_public')
import baseline as B

DATA = Path('data').resolve()
CLASSES = json.loads((DATA / 'classes.json').read_text())
MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
# PROTO_PIL=1: emulate PIL's uint8 bilinear resize (horizontal pass, round, vertical pass, round), as torchvision does
# on the official PIL images; matches PIL on 98.7% of output pixels (mean |diff| 0.013 level) vs 0.28 level for the
# float path. Rounding is straight-through so PFD can differentiate through it.
PIL_ROUND = os.environ.get('PROTO_PIL', '0') == '1'


def _st_round(v):
    return v + (v.round().clamp(0, 255) - v).detach()


def resize_bilinear(x01, size):
    """x01 float NCHW in [0,1] -> size (h, w), PIL-faithful when PIL_ROUND."""
    if not PIL_ROUND:
        return F.interpolate(x01, size=size, mode='bilinear', align_corners=False, antialias=True)
    v = x01 * 255
    if v.shape[-1] != size[1]:
        v = _st_round(F.interpolate(v, size=(v.shape[-2], size[1]), mode='bilinear', align_corners=False, antialias=True))
    if v.shape[-2] != size[0]:
        v = _st_round(F.interpolate(v, size=size, mode='bilinear', align_corners=False, antialias=True))
    return v / 255

def load_split(split, cache=Path('cache')):
    cache.mkdir(exist_ok=True)
    f = cache / f'{split}.pt'
    if f.exists():
        return torch.load(f)
    rows = list(csv.DictReader(open(DATA / f'{split}.csv')))
    x = torch.from_numpy(np.stack([np.asarray(Image.open(DATA / split / r['filename']).convert('RGB')) for r in rows]))
    y = torch.tensor([CLASSES.index(r['label']) for r in rows]) if split != 'test' else None
    torch.save((x, y), f)
    return x, y

def to_eval(xu8):
    """uint8 NHWC 64x64 -> normalized 224 (Resize 256 bilinear + center crop 224), GPU tensor."""
    x = xu8.permute(0, 3, 1, 2).float().div(255)
    x = resize_bilinear(x, (256, 256))
    x = x[:, :, 16:240, 16:240]
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def augment(x01, gen=None):
    """x01: float NCHW in [0,1] at any res. RandomResizedCrop(224, scale .75-1, ratio 3/4-4/3) + flips, differentiable."""
    n, _, h, w = x01.shape
    out = []
    for i in range(n):
        area = h * w
        for _ in range(10):
            s = random.uniform(0.75, 1.0) * area
            r = np.exp(random.uniform(np.log(3 / 4), np.log(4 / 3)))
            cw, ch = int(round((s * r) ** .5)), int(round((s / r) ** .5))
            if 0 < cw <= w and 0 < ch <= h:
                break
        else:
            cw, ch = w, h
        top, left = random.randint(0, h - ch), random.randint(0, w - cw)
        crop = x01[i:i + 1, :, top:top + ch, left:left + cw]
        crop = resize_bilinear(crop, (224, 224))
        if random.random() < .5: crop = crop.flip(3)
        if random.random() < .5: crop = crop.flip(2)
        out.append(crop)
    x = torch.cat(out)
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def augment_views(x01, views):
    """Same as augment() but with given views: rows (top, left, h, w, hflip, vflip) as recorded by record_views.py
    from baseline.py's own DataLoader (exact official crops and flips)."""
    out = []
    for k in range(len(x01)):
        i, j, h, w, hf, vf = [int(t) for t in views[k]]
        crop = resize_bilinear(x01[k:k + 1, :, i:i + h, j:j + w], (224, 224))
        if hf: crop = crop.flip(3)
        if vf: crop = crop.flip(2)
        out.append(crop)
    x = torch.cat(out)
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def fresh_model(seed, device='cuda'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = B.build_model(len(CLASSES)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    return m, opt, sch

@torch.no_grad()
def evaluate(m, xu8, y, bs=256):
    m.eval(); correct = 0
    for i in range(0, len(xu8), bs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = m(to_eval(xu8[i:i + bs].cuda())).argmax(1)
        correct += (p.cpu() == y[i:i + bs]).sum().item()
    m.train(); return correct / len(xu8)
