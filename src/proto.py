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
# PROTO_EXACT=1: training batches built exactly like baseline.py's DataLoader (PIL crop/resize/flips of the 8-bit images,
# ToTensor and Normalize on the CPU), with the gradient taken through the differentiable copy (straight-through), and
# validation through the official transforms in fp32. The replay then matches the official run (checked: 84.35 and
# 83.93 for a 64x64 and a 224x224 set, same as baseline.py).
EXACT = os.environ.get('PROTO_EXACT', '0') == '1'
if EXACT:                      # many tiny CPU tensors per batch: a big thread pool only spins
    torch.set_num_threads(min(8, torch.get_num_threads()))
_SPLIT_PTR = {}


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
        x, y = torch.load(f)
        _SPLIT_PTR[split] = x.data_ptr()
        return x, y
    rows = list(csv.DictReader(open(DATA / f'{split}.csv')))
    x = torch.from_numpy(np.stack([np.asarray(Image.open(DATA / split / r['filename']).convert('RGB')) for r in rows]))
    y = torch.tensor([CLASSES.index(r['label']) for r in rows]) if split != 'test' else None
    torch.save((x, y), f)
    _SPLIT_PTR[split] = x.data_ptr()
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

_NMEAN, _NSTD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


def _exact_batch(x01, views):
    """The official training batch, bit for bit: 8-bit images, PIL crop + bilinear resize + flips, ToTensor and
    Normalize on the CPU (as baseline.py's DataLoader workers do), then moved to x01's device."""
    from torchvision.transforms import functional as TF
    u8 = (x01.detach().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    out = []
    for k in range(len(u8)):
        i, j, h, w, hf, vf = [int(t) for t in views[k]]
        im = TF.resized_crop(Image.fromarray(u8[k], 'RGB'), i, j, h, w, [224, 224],
                             interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        if hf: im = TF.hflip(im)
        if vf: im = TF.vflip(im)
        out.append(TF.normalize(TF.to_tensor(im), _NMEAN, _NSTD))
    return torch.stack(out).to(x01.device)


def augment_views(x01, views):
    """Same as augment() but with given views: rows (top, left, h, w, hflip, vflip) as recorded by record_views.py
    from baseline.py's own DataLoader (exact official crops and flips). With PROTO_EXACT=1 the values are the official
    ones and the gradient is the one of the differentiable copy."""
    emul = _augment_views_emul(x01, views)
    if not EXACT:
        return emul
    ex = _exact_batch(x01, views)
    return ex + (emul - emul.detach()) if emul.requires_grad else ex


def _augment_views_emul(x01, views):
    out = []
    for k in range(len(x01)):
        i, j, h, w, hf, vf = [int(t) for t in views[k]]
        crop = resize_bilinear(x01[k:k + 1, :, i:i + h, j:j + w], (224, 224))
        if hf: crop = crop.flip(3)
        if vf: crop = crop.flip(2)
        out.append(crop)
    x = torch.cat(out)
    return (x - MEAN.to(x.device)) / STD.to(x.device)

def pil_views_u8(x01, views):
    """The official training inputs, exactly: torchvision's PIL crop + bilinear resize + flips of the uint8 images.
    Returns float NCHW in [0, 1] on x01's device (no gradient)."""
    from torchvision.transforms import functional as TF
    u8 = (x01.detach().clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).cpu().numpy()
    out = []
    for k in range(len(u8)):
        i, j, h, w, hf, vf = [int(t) for t in views[k]]
        im = TF.resized_crop(Image.fromarray(u8[k], 'RGB'), i, j, h, w, [224, 224],
                             interpolation=TF.InterpolationMode.BILINEAR, antialias=True)
        if hf: im = TF.hflip(im)
        if vf: im = TF.vflip(im)
        out.append(torch.from_numpy(np.asarray(im).copy()).permute(2, 0, 1))
    return torch.stack(out).float().div(255).to(x01.device)


def augment_views_exact(x01, views):
    """augment_views whose forward value is the exact PIL result (straight-through: the gradient is the one of the
    differentiable emulation). Also rounds the input to 8 bits in the forward pass, as the PNG files do."""
    emul = augment_views(x01, views) * STD.to(x01.device) + MEAN.to(x01.device)
    exact = pil_views_u8(x01, views)
    x = emul + (exact - emul).detach()
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def fresh_model(seed, device='cuda'):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    m = B.build_model(len(CLASSES)).to(device)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-5, weight_decay=0.05)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=5)
    return m, opt, sch

_VAL = {}


def _exact_val():
    """Validation inputs exactly as baseline.py makes them (PIL + WEIGHTS.transforms()), cached on the CPU."""
    if not _VAL:
        f = Path('cache/validation_exact.pt')    # 2.4 GB, made once
        if f.exists():
            x = torch.load(f)
        else:
            rows = list(csv.DictReader(open(DATA / 'validation.csv')))
            tf = B.WEIGHTS.transforms()
            x = torch.stack([tf(Image.open(DATA / 'validation' / r['filename']).convert('RGB')) for r in rows])
            torch.save(x, f)
        _VAL['x'] = x.pin_memory()
    return _VAL['x']


@torch.no_grad()
def evaluate(m, xu8, y, bs=256):
    if EXACT and xu8.data_ptr() == _SPLIT_PTR.get('validation'):
        xv = _exact_val(); m.eval(); correct = 0
        for i in range(0, len(xv), bs):
            correct += (m(xv[i:i + bs].cuda(non_blocking=True)).argmax(1).cpu() == y[i:i + bs]).sum().item()
        m.train(); return correct / len(xv)
    m.eval(); correct = 0
    for i in range(0, len(xu8), bs):
        with torch.autocast('cuda', dtype=torch.bfloat16):
            p = m(to_eval(xu8[i:i + bs].cuda())).argmax(1)
        correct += (p.cpu() == y[i:i + bs]).sum().item()
    m.train(); return correct / len(xu8)
