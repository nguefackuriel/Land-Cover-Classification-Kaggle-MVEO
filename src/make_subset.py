"""Build a challenge-format data dir from a list of (src_path, label) training images.
Validation/test are symlinked to the real data so the official baseline.py runs unchanged."""
import argparse, csv, os, random, shutil
from pathlib import Path

DATA = Path('data').resolve()

def build(out, items):
    out = Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / 'train').mkdir(parents=True)
    for name in ['validation', 'test', 'validation.csv', 'test.csv', 'classes.json']:
        os.symlink(DATA / name, out / name)
    with open(out / 'train.csv', 'w', newline='') as f:
        w = csv.writer(f); w.writerow(['filename', 'label'])
        for i, (src, label) in enumerate(items):
            name = f'syn_{i:05d}.png'
            shutil.copy(src, out / 'train' / name)
            w.writerow([name, label])

def random_real(n_per_class, seed, counts=None):
    rows = list(csv.DictReader(open(DATA / 'train.csv')))
    by = {}
    for r in rows:
        by.setdefault(r['label'], []).append(r['filename'])
    rng = random.Random(seed)
    items = []
    for c in sorted(by):
        k = counts[c] if counts else n_per_class
        items += [(DATA / 'train' / f, c) for f in rng.sample(by[c], k)]
    return items

if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--ipc', type=int, required=True)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    build(a.out, random_real(a.ipc, a.seed))
