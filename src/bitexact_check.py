"""Can the replica be bit-exact with baseline.py? Replay a set's seed-42 run with training batches built exactly like the
official DataLoader (PIL crop/resize/flips, ToTensor, Normalize on the CPU, then .to(cuda)) and compare the final
weights with an official best.pt. usage: bitexact_check.py <set_dir> <views.pt> <official_best.pt>"""
import sys, json, csv, torch, numpy as np, torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.transforms import functional as TF
import proto as P
set_dir, vpath, off = sys.argv[1], sys.argv[2], sys.argv[3]
rows = list(csv.DictReader(open(f'{set_dir}/train.csv')))
imgs = [Image.open(f'{set_dir}/train/{r["filename"]}').convert('RGB') for r in rows]
labels = torch.tensor([P.CLASSES.index(r['label']) for r in rows])
V = torch.load(vpath)
norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])


def batch(idx, vw):
    out = []
    for k, i_ in enumerate(idx.tolist()):
        i, j, h, w, hf, vf = [int(t) for t in vw[k]]
        im = TF.resized_crop(imgs[i_], i, j, h, w, [224, 224], interpolation=transforms.InterpolationMode.BILINEAR, antialias=True)
        if hf: im = TF.hflip(im)
        if vf: im = TF.vflip(im)
        out.append(norm(TF.to_tensor(im)))
    return torch.stack(out)


m, opt, sch = P.fresh_model(V['seed'])
for batches in V['epochs']:
    for idx, vw in batches:
        xb = batch(idx, vw).cuda(); yb = labels[idx].cuda()
        opt.zero_grad(set_to_none=True); F.cross_entropy(m(xb), yb).backward(); opt.step()
    sch.step()
o = torch.load(off, map_location='cpu', weights_only=True)['model_state']
a = {k: v.detach().cpu() for k, v in m.state_dict().items()}
d = torch.cat([(o[k].float() - a[k].float()).flatten() for k in a])
print(json.dumps({'set': set_dir, 'max_abs_diff': d.abs().max().item(), 'frac_nonzero': (d != 0).float().mean().item()}))

# validation exactly as baseline.py: PIL images through WEIGHTS.transforms(), batches of 32, fp32
import baseline as Bl
val = Bl.Images(P.DATA, 'validation', P.CLASSES)
loader = torch.utils.data.DataLoader(val, batch_size=32, shuffle=False, num_workers=2)
m.eval(); correct = total = 0
with torch.no_grad():
    for xb, yb, _ in loader:
        correct += (m(xb.cuda()).argmax(1).cpu() == yb).sum().item(); total += len(yb)
print(json.dumps({'set': set_dir, 'val_exact_pipeline': round(correct / total, 4)}))
