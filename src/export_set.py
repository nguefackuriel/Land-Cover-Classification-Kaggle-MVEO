"""Write a challenge-format data dir from saved train indices (cache/*.pt) using the original PNG files."""
import csv, sys, torch
from pathlib import Path
import make_subset as M

DATA = Path('data').resolve()
rows = list(csv.DictReader(open(DATA / 'train.csv')))
idx = torch.load(sys.argv[1]).tolist()
M.build(sys.argv[2], [(DATA / 'train' / rows[i]['filename'], rows[i]['label']) for i in idx])
print(len(idx), 'images ->', sys.argv[2])
