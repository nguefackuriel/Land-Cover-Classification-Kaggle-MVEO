"""Make a challenge-format data folder from a set of training images (a folder with train.csv and its PNGs, e.g.
submissions/<name>): train/ and train.csv are copied as they are (same files, same row order), validation/, test/,
their csv files and classes.json are linked to data/. baseline.py can then be run on it unchanged.
usage: python src/build_set.py submissions/2026-10-15_n10_exact42_stage2 sets/final"""
import csv, os, shutil, sys
from pathlib import Path

src, out = Path(sys.argv[1]), Path(sys.argv[2])
data = Path('data').resolve()
if out.exists():
    shutil.rmtree(out)
(out / 'train').mkdir(parents=True)
rows = list(csv.DictReader(open(src / 'train.csv')))
img = src / 'train' if (src / 'train').is_dir() else src
for r in rows:
    shutil.copy(img / r['filename'], out / 'train' / r['filename'])
shutil.copy(src / 'train.csv', out / 'train.csv')
for name in ['validation', 'test', 'validation.csv', 'test.csv', 'classes.json']:
    os.symlink(data / name, out / name)
print(f'{len(rows)} images -> {out}')
