"""Project a 224x224 set onto 64x64 images, to start the 64 track from the 224 solution.
Three versions: PIL box downsample, PIL bilinear (antialiased) downsample, and a least-squares fit that makes the
64 images' official training views (64 views file, crop + upsample) match the 224 set's official views as closely as
possible (all 5 steps, Adam on the 64 pixels, then rounded to 8 bits).
usage: proj64.py <set224.pt> <views64.pt> <views224.pt> <out prefix>   -> <prefix>_box.pt, _bil.pt, _ls.pt"""
import sys, torch, numpy as np
from PIL import Image
import proto as P

src, v64, v224, out = sys.argv[1:5]
d = torch.load(src, weights_only=False)
x224, y = d['x01'].float(), d['y']
V64 = torch.load(v64, weights_only=False); V224 = torch.load(v224, weights_only=False)


def pil_down(x, mode):
    u8 = (x.clamp(0, 1) * 255).round().byte().permute(0, 2, 3, 1).numpy()
    r = [np.asarray(Image.fromarray(a, 'RGB').resize((64, 64), mode)) for a in u8]
    return torch.from_numpy(np.stack(r)).permute(0, 3, 1, 2).float().div(255)


for name, mode in (('box', Image.BOX), ('bil', Image.BILINEAR)):
    torch.save({'x01': pil_down(x224, mode), 'y': y, 'src': src}, f'{out}_{name}.pt')

# least squares on the official views: target = exact 224 views, model = differentiable 64 views
dev = 'cuda'
pairs = []
for (i64, w64), (i224, w224) in zip([b for e in V64['epochs'] for b in e], [b for e in V224['epochs'] for b in e]):
    assert torch.equal(i64, i224)
    with torch.no_grad():
        t = P.pil_views_u8(x224[i224].to(dev), w224)
    pairs.append((i64.to(dev), w64, t))
x = pil_down(x224, Image.BILINEAR).to(dev).requires_grad_(True)
opt = torch.optim.Adam([x], lr=2e-3)
for it in range(400):
    opt.zero_grad()
    loss = 0
    for idx, w, t in pairs:
        e = P._augment_views_emul(x[idx], w) * P.STD.to(dev) + P.MEAN.to(dev)
        loss = loss + (e - t).pow(2).mean()
    loss.backward(); opt.step()
    with torch.no_grad():
        x.clamp_(0, 1)
    if it % 100 == 0 or it == 399:
        print('ls it', it, 'rmse (levels)', (loss.item() / len(pairs)) ** .5 * 255, flush=True)
xq = (x.detach().cpu() * 255).round().div(255)
torch.save({'x01': xq, 'y': y, 'src': src}, f'{out}_ls.pt')
print('saved', out)
