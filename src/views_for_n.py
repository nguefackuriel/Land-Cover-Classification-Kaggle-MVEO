"""Record the exact seed-<s> official views (batches, crops, flips) for a set of n 64x64 images. The views depend only
on n and the image size, so a dummy set is enough. usage: views_for_n.py <n> <seed> [size]  -> cache/views_n<n>_s<seed>.pt"""
import csv, json, os, subprocess, sys, tempfile
from pathlib import Path
import numpy as np
from PIL import Image
DATA = Path('data').resolve()
W = Path('.')
n, seed = int(sys.argv[1]), int(sys.argv[2])
size = int(sys.argv[3]) if len(sys.argv) > 3 else 64
classes = json.loads((DATA / 'classes.json').read_text())
with tempfile.TemporaryDirectory() as d:
    d = Path(d); (d / 'train').mkdir()
    for nm in ['validation', 'test', 'validation.csv', 'test.csv', 'classes.json']:
        os.symlink(DATA / nm, d / nm)
    with open(d / 'train.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['filename', 'label'])
        for i in range(n):
            fn = f'distilled_{i:04d}.png'
            Image.fromarray(np.zeros((size, size, 3), np.uint8), 'RGB').save(d / 'train' / fn)
            w.writerow([fn, classes[i % 10]])
    suffix = '' if size == 64 else f'_sz{size}'
    out = W / 'cache' / f'views_n{n}_s{seed}{suffix}.pt'
    subprocess.run([sys.executable, str(Path(__file__).with_name('record_views.py')), str(d), str(seed), str(out)], check=True)
