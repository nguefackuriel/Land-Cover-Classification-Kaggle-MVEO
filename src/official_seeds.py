"""Unmodified baseline.py on a set for several seeds; prints best val per seed, mean and std. Deletes each run's
343 MB best.pt after reading history.json (only inside the run dirs this script creates).
usage: official_seeds.py <set_dir> <name> 0 1 2 3 4"""
import json, shutil, statistics, subprocess, sys
from pathlib import Path
W = Path('.')
BASELINE = 'ext/challenge_2026_public/baseline.py'
set_dir, name, seeds = sys.argv[1], sys.argv[2], [int(s) for s in sys.argv[3:]]
best = {}
for s in seeds:
    run = W / 'runs' / f'off_{name}_s{s}'
    assert not run.exists(), run
    with open(W / 'runs' / f'off_{name}_s{s}.log', 'w') as log:
        subprocess.run([sys.executable, BASELINE, 'train', '--data', set_dir, '--output', str(run),
                        '--seed', str(s)], stdout=log, stderr=subprocess.STDOUT, check=True)
    hist = json.loads((run / 'history.json').read_text())
    best[s] = max(h['validation_accuracy'] for h in hist)
    (run / 'best.pt').unlink(missing_ok=True)
    print(json.dumps({'set': name, 'seed': s, 'best_val': round(best[s], 4)}), flush=True)
v = list(best.values())
print(json.dumps({'set': name, 'seeds': seeds, 'mean': round(statistics.mean(v), 4),
                  'std': round(statistics.stdev(v), 4) if len(v) > 1 else 0}), flush=True)
