"""Train with the unmodified official baseline.py (all defaults, seed 42) on a distilled set, then write a
Kaggle-ready CSV with the __num_train_images__ sentinel and validate it.
usage: make_submission.py <set_dir> <out_name>"""
import csv, json, subprocess, sys
from pathlib import Path

W = Path('.')
BASELINE = 'ext/challenge_2026_public/baseline.py'
PY = sys.executable
set_dir, name = Path(sys.argv[1]), sys.argv[2]
run = W / 'runs' / f'sub_{name}'
log = open(W / 'runs' / f'sub_{name}.log', 'w')
subprocess.run([PY, BASELINE, 'train', '--data', str(set_dir), '--output', str(run)], stdout=log, stderr=subprocess.STDOUT, check=True)

n = sum(1 for _ in csv.DictReader(open(set_dir / 'train.csv')))
assert n == len(list((set_dir / 'train').iterdir())), 'train.csv and train/ disagree'
preds = list(csv.DictReader(open(run / 'submission.csv')))
test = [r['filename'] for r in csv.DictReader(open(set_dir / 'test.csv'))]
classes = json.loads((set_dir / 'classes.json').read_text())
assert [p['filename'] for p in preds] == test and all(p['label'] in classes for p in preds)

out = W / 'submissions' / f'{name}.csv'; out.parent.mkdir(exist_ok=True)
with open(out, 'w', newline='') as f:
    w = csv.writer(f); w.writerow(['filename', 'label']); w.writerow(['__num_train_images__', n])
    w.writerows((p['filename'], p['label']) for p in preds)
rows = list(csv.reader(open(out)))
assert len(rows) == 4052 and len({r[0] for r in rows[1:]}) == 4051
hist = json.loads((run / 'history.json').read_text())
best = max(h['validation_accuracy'] for h in hist)
print(json.dumps({'file': str(out), 'n': n, 'val_per_epoch': [round(h['validation_accuracy'], 4) for h in hist],
                  'best_val': round(best, 4)}))
