"""Write distilled images (cache/pfd_*_best.pt: x01 float NCHW quantized to /255, y) as 64x64 RGB PNGs in a
challenge-format data dir, then check the PNGs read back bit-exact.
usage: export_synth.py cache/pfd_v2_best.pt sets/pfd_v2"""
import csv, json, os, shutil, sys
from pathlib import Path
import numpy as np, torch
from PIL import Image

DATA = Path('data').resolve()
src, out = sys.argv[1], Path(sys.argv[2])
d = torch.load(src)
x = (d['x01'].clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
classes = json.loads((DATA / 'classes.json').read_text())
if out.exists():
    shutil.rmtree(out)
(out / 'train').mkdir(parents=True)
for name in ['validation', 'test', 'validation.csv', 'test.csv', 'classes.json']:
    os.symlink(DATA / name, out / name)
with open(out / 'train.csv', 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['filename', 'label'])
    for i, (img, lab) in enumerate(zip(x, d['y'].tolist())):
        fn = f'distilled_{i:04d}.png'
        Image.fromarray(img, 'RGB').save(out / 'train' / fn)
        assert np.array_equal(np.asarray(Image.open(out / 'train' / fn)), img)
        w.writerow([fn, classes[lab]])
print(f'{len(x)} images -> {out} (from it={d.get("it")}, accs={d.get("accs")})')
