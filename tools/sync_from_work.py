"""Copy our working folder into the repo, so nothing is lost if the machine goes down. Run from the repo root:

    python tools/sync_from_work.py [--work ../work] [--dry-run]

What it copies:
- scripts: work/*.py into src/, with absolute paths made relative to the repo root. A script is copied only when the
  work copy is newer than the src copy (some src files have small fixes of their own, those are kept);
- reusable shell pipelines (loop_*.sh) into src/, other shell files into results/logs/ as a record;
- run logs (work/runs/*.jsonl, *.out, *.log) into results/logs/, and the history.json of official runs into
  results/official_runs/;
- small checkpoints (image sets, recorded views, starting sets) into checkpoints/; network state dicts (343 MB) stay out,
  the scripts can make them again;
- each new Kaggle submission listed in tools/submissions.json into submissions/<folder>/ (PNGs, train.csv, sha256,
  Kaggle csv, scores.json);
- the planning notes (../PLAN.md) into notes/PLAN.md.
"""
import argparse, hashlib, json, os, re, shutil
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument('--work', default='../work')
ap.add_argument('--plan', default='../PLAN.md')
ap.add_argument('--dry-run', action='store_true')
a = ap.parse_args()
W = Path(a.work).resolve()
ROOT = Path('.').resolve()
assert (ROOT / 'src').is_dir() and (ROOT / '.git').exists(), 'run from the repo root'
ABS = '/home/ubuntu/Uriel/data_centric'
copied = []


def put(src, dst, text=None):
    dst = ROOT / dst
    if text is None:
        if dst.exists() and dst.stat().st_size == src.stat().st_size and dst.read_bytes() == src.read_bytes():
            return
    elif dst.exists() and dst.read_text() == text:
        return
    copied.append(str(dst.relative_to(ROOT)))
    if a.dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if text is None:
        shutil.copy2(src, dst)
    else:
        dst.write_text(text)


def py_rewrite(s):
    s = s.replace(f"sys.path.insert(0, '{ABS}/work')\n", '')
    s = re.sub(rf"{ABS}/work/(\w+\.py)", r'src/\1', s)
    s = s.replace(f"Path('{ABS}/work')", "Path('.')")
    s = s.replace(f'{ABS}/work/', '')
    s = s.replace(f"'{ABS}/ext/challenge_2026_public", "'ext/challenge_2026_public")
    s = s.replace(f"Path('{ABS}/data')", "Path('data').resolve()")
    s = s.replace("'/home/ubuntu/vllm-env/bin/python'", 'sys.executable')
    return s


def sh_rewrite(s):
    s = s.replace(f'cd {ABS}/work', 'cd "$(dirname "$0")/.."')
    s = s.replace('PY=/home/ubuntu/vllm-env/bin/python', 'PY=${PY:-python}')
    s = re.sub(r'\$PY (\w+\.py)', r'$PY src/\1', s)
    s = s.replace(f'{ABS}/ext/challenge_2026_public', 'ext/challenge_2026_public')
    s = re.sub(r'runs/(loop_\w+\.sh)', r'src/\1', s)
    return s


# 1. python scripts
for f in sorted(W.glob('*.py')):
    dst = ROOT / 'src' / f.name
    if dst.exists() and dst.stat().st_mtime >= f.stat().st_mtime:
        continue
    put(f, f'src/{f.name}', py_rewrite(f.read_text()))

# 2. shell files
for f in sorted((W / 'runs').glob('*.sh')):
    if f.name.startswith('loop_'):
        put(f, f'src/{f.name}', sh_rewrite(f.read_text()))
    else:
        put(f, f'results/logs/{f.name}')

# 3. logs and official run histories
for f in sorted((W / 'runs').iterdir()):
    if f.is_file() and f.suffix in ('.jsonl', '.out', '.log') and f.stat().st_size < 20_000_000:
        put(f, f'results/logs/{f.name}')
    elif f.is_dir() and (f / 'history.json').exists() and f.name.startswith(('off_', 'sub_', 'recheck_')):
        put(f / 'history.json', f'results/official_runs/{f.name}/history.json')

# 4. small checkpoints
try:
    import numpy as np, torch
    from PIL import Image
    TORCH = True
except ImportError:                            # cron may run without torch: then the big sets are skipped
    TORCH = False
CLASSES = json.loads((ROOT / 'data' / 'classes.json').read_text()) if (ROOT / 'data' / 'classes.json').exists() else \
    json.loads((W.parent / 'data' / 'classes.json').read_text())


def put_png_set(f, out):
    """Write a .pt image set (x01 float, y) as train/*.png + train.csv (what dd_lib.load_set reads back)."""
    d = torch.load(f, map_location='cpu')
    x = (d['x01'].clamp(0, 1) * 255).round().to(torch.uint8).permute(0, 2, 3, 1).numpy()
    rows = 'filename,label\n' + ''.join(f'distilled_{i:04d}.png,{CLASSES[int(l)]}\n' for i, l in enumerate(d['y'].tolist()))
    dst = ROOT / out
    if (dst / 'train.csv').exists() and (dst / 'source_mtime').exists() and \
            (dst / 'source_mtime').read_text() == str(int(f.stat().st_mtime)):
        return
    copied.append(out)
    if a.dry_run:
        return
    (dst / 'train').mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(x):
        Image.fromarray(img, 'RGB').save(dst / 'train' / f'distilled_{i:04d}.png')
    (dst / 'train.csv').write_text(rows)
    (dst / 'source_mtime').write_text(str(int(f.stat().st_mtime)))

keep = re.compile(r'^(init_\w+|views_\w+|sm_\w+_best|sm_hyb3_it120|cd_\w+_best|pfd3_pf_\w+_best|pfd3_n10_exact_med_best'
                  r'|pfd3_n10_exact_s2_it280|pfd3_sz_n10_best)\.pt$')
for f in sorted((W / 'cache').glob('*.pt')):
    if not keep.match(f.name) or f.name.startswith(('ceiling', 'state_')) or f.stat().st_size > 20_000_000:
        continue
    if f.stat().st_size < 2_000_000:          # 64x64 sets, views, inits: as they are
        put(f, f'checkpoints/{f.name}')
    elif TORCH:                                # 224x224 sets (6 MB as float): 8-bit PNG folders, about 1 MB
        put_png_set(f, f'checkpoints/png/{f.stem}')

# 5. submissions
subs = json.loads((ROOT / 'tools' / 'submissions.json').read_text())
for s in subs:
    out = Path('submissions') / s['folder']
    imgs = W / 'submissions' / f"{s['name']}_images"
    for png in sorted(imgs.glob('*.png')):
        put(png, out / 'train' / png.name)
    put(imgs / 'train.csv', out / 'train.csv')
    sums = ''.join(f"{hashlib.sha256(p.read_bytes()).hexdigest()}  {p.name}\n" for p in sorted(imgs.glob('*.png')))
    put(None, out / 'sha256.txt', sums)
    put(W / 'submissions' / f"{s['name']}.csv", out / 'kaggle_submission.csv')
    put(None, out / 'scores.json', json.dumps({'n_images': s['n'], 'official_val_seed42': s['val'],
                                              'kaggle_public_score': s['score']}, indent=1) + '\n')

# 6. planning notes
plan = Path(a.plan)
if plan.exists():
    put(plan.resolve(), 'notes/PLAN.md')

print(f"{'would copy' if a.dry_run else 'copied'} {len(copied)} files")
for c in copied[:400]:
    print('  ', c)
