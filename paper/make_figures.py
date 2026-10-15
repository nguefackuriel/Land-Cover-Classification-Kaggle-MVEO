"""Draw the figures of the paper from the logs in results/ and the images in submissions/ and data/.
usage (from the repo root): python paper/make_figures.py        (needs matplotlib, numpy, pillow)"""
import json, math, re
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image

LOGS, RUNS, OUT = Path('results/logs'), Path('results/official_runs'), Path('paper/figures')
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['DejaVu Serif'], 'mathtext.fontset': 'dejavuserif',
                     'font.size': 8, 'axes.labelsize': 8, 'legend.fontsize': 7, 'xtick.labelsize': 7,
                     'ytick.labelsize': 7, 'axes.linewidth': 0.6, 'lines.linewidth': 1.2, 'savefig.bbox': 'tight',
                     'savefig.pad_inches': 0.02})
ONE, TWO = 3.45, 7.1           # IEEE column and page widths (inches)
NAMES = ['AnnualCrop', 'Forest', 'HerbVeg', 'Highway', 'Industrial', 'Pasture', 'PermCrop', 'Residential', 'River',
         'SeaLake']
SIZE = lambda n: 0.5 * (1 - math.log(n) / math.log(18900))


def jsonl(name):
    out = []
    for line in open(LOGS / name):
        line = line.strip()
        if line.startswith('{'):
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass
    return out


def evals(name):
    return [r for r in jsonl(name) if 'val_accs' in r]


def best_val(run):
    return max(h['validation_accuracy'] for h in json.load(open(RUNS / run / 'history.json')))


# ---------------------------------------------------------------- Fig. 1: score landscape
def fig_score():
    fig, ax = plt.subplots(figsize=(ONE, 2.35))
    n = np.logspace(math.log10(8), math.log10(200), 400)
    acc = np.linspace(0.3, 1.0, 400)
    N, A = np.meshgrid(n, acc)
    S = 0.5 * A + 0.5 * (1 - np.log(N) / math.log(18900))
    levels = [0.58, 0.62, 0.66, 0.70, 0.74, 0.78]
    cs = ax.contour(N, A * 100, S, levels=levels, colors='0.6', linewidths=0.5)
    ax.clabel(cs, fmt='%.2f', fontsize=6, inline_spacing=2,
              manual=[(45, 200 * (lv - SIZE(45))) for lv in levels])
    for k in range(1, 7):                                   # the step staircase: 5 * ceil(n / 32) steps
        lo, hi = max(32 * (k - 1) + 0.5, 8), min(32 * k + 0.5, 200)
        ax.axvspan(lo, hi, color='0.93' if k % 2 else 'white', zorder=0)
        if k <= 4:
            ax.text(math.sqrt(lo * hi), 32, f'{5 * k}', ha='center', fontsize=6, color='0.35')
    ax.text(8.6, 36, 'training steps:', ha='left', fontsize=6, color='0.35')
    pts = [(120, 87.19, 'o', '0.45', 'real images (medoids)'),
           (120, 89.94, 's', 'C0', 'PFD, mean over seeds'),
           (10, 61.2, 's', 'C0', None),
           (10, 80.62, '*', 'C3', 'PFD, seed-42 run only'),
           (10, 52.24, 'v', 'C3', 'same set, seeds 0-4')]
    for x, y, m, c, lab in pts:
        ax.plot(x, y, m, color=c, ms=6 if m == '*' else 4, label=lab, zorder=3)
    ax.set_xscale('log')
    ax.set_xticks([10, 20, 32, 64, 96, 128, 200])
    ax.set_xticklabels(['10', '20', '32', '64', '96', '128', '200'])
    ax.set_xlim(8, 200); ax.set_ylim(30, 100)
    ax.set_xlabel('number of training images $n$'); ax.set_ylabel('validation accuracy (%)')
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.2), ncol=2, frameon=False, handletextpad=0.3,
              columnspacing=1.0)
    fig.savefig(OUT / 'score_landscape.pdf'); plt.close(fig)


# ---------------------------------------------------------------- Fig. 2: frontier
def fig_frontier():
    med = jsonl('frontier_medoids.out')
    fig, ax = plt.subplots(figsize=(ONE, 2.3))
    ns = [r['n'] for r in med]; m = [100 * r['mean'] for r in med]; sd = [100 * np.std(r['accs']) for r in med]
    ax.errorbar(ns, m, yerr=sd, fmt='o-', color='0.45', ms=3, capsize=1.5, lw=0.9, label='real images (k-means medoids)')
    rnd = {}
    for line in open(LOGS / 'select_eval.txt'):
        g = re.match(r'ipc=(\d+) n=(\d+) (\{.*\})', line)
        if g:
            rnd[int(g.group(2))] = json.loads(g.group(3))['random']
    xs = sorted(rnd)
    ax.plot(xs, [100 * np.mean(rnd[x]) for x in xs], 'x--', color='0.6', ms=4, lw=0.8, label='random real images')
    pfd = [(10, 61.2), (40, 73.7), (64, 78.38), (96, 86.03), (120, 89.94)]
    ax.plot(*zip(*pfd), 's-', color='C0', ms=3.5, label='PFD (all seeds)')
    for (a, b), acc in zip([(9, 32.5), (32.5, 64.5), (64.5, 128.5)], [60.4, 80.1, 91.7]):
        ax.plot([a, b], [acc, acc], color='C2', lw=1.0, ls=':')
    ax.plot([], [], color='C2', ls=':', label='oracle: 1024 real images per step')
    for x in (32.5, 64.5, 96.5, 128.5):
        ax.axvline(x, color='0.85', lw=0.6, zorder=0)
    ax.set_xscale('log'); ax.set_xlim(9, 175); ax.set_ylim(16, 96)
    ax.set_xticks([10, 20, 32, 64, 96, 128, 160]); ax.set_xticklabels(['10', '20', '32', '64', '96', '128', '160'])
    ax.set_xlabel('number of training images $n$'); ax.set_ylabel('validation accuracy (%)')
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(OUT / 'frontier.pdf'); plt.close(fig)


# ---------------------------------------------------------------- Fig. 3: optimization curves at n = 120
def fig_curves120():
    fig, ax = plt.subplots(figsize=(ONE, 2.2))
    ce = evals('pfd3_n120.jsonl')
    ax.plot([r['it'] for r in ce], [100 * r['mean'] for r in ce], 'o-', ms=2.5, color='C1', label='cross-entropy')
    t1 = evals('pfd3_n120_T0_fromopt.jsonl'); t2 = evals('pfd3_n120_T0_stage2.jsonl')
    xs = [250 + r['it'] for r in t1] + [350 + r['it'] for r in t2[1:]]
    ys = [100 * r['mean'] for r in t1] + [100 * r['mean'] for r in t2[1:]]
    ax.plot(xs, ys, 's-', ms=2.5, color='C0', label='scale-free (from step 250)')
    med = evals('pfd3_n120_T0_med.jsonl')
    ax.plot([r['it'] for r in med], [100 * r['mean'] for r in med], 's--', ms=2.5, color='C0', alpha=0.5,
            label='scale-free from the medoids')
    ax.axvline(250, color='0.8', lw=0.6, zorder=0)
    ax.set_xlabel('outer step'); ax.set_ylabel('validation accuracy (%)\nmean of 3 seeds, replica')
    ax.legend(loc='lower right', frameon=False)
    fig.savefig(OUT / 'curves_n120.pdf'); plt.close(fig)


# ---------------------------------------------------------------- Fig. 4: per-seed official accuracy
def fig_seeds():
    fig, axs = plt.subplots(1, 2, figsize=(TWO, 2.2), gridspec_kw={'width_ratios': [1.25, 1]})
    seeds = [42, 0, 1, 2, 3, 4]
    sets120 = [('real images (medoids)', 'sub_medoids_n120', 'off_medoids_n120', '0.5'),
               ('PFD, cross-entropy', 'sub_pfd3_n120', 'off_pfd3_n120', 'C1'),
               ('PFD, scale-free', 'off_pfd3_T0s2_s42', 'off_pfd3_T0s2', 'C0')]
    w = 0.26
    for k, (lab, s42, base, c) in enumerate(sets120):
        vals = [best_val(s42 if s == 42 else f'{base}_s{s}') for s in seeds]
        axs[0].bar(np.arange(6) + (k - 1) * w, [100 * v for v in vals], w, color=c, label=lab)
    axs[0].set_xticks(range(6)); axs[0].set_xticklabels([str(s) for s in seeds])
    axs[0].set_ylim(84, 92); axs[0].set_xlabel('seed of baseline.py'); axs[0].set_ylabel('validation accuracy (%)')
    axs[0].set_title('$n = 120$, built for all seeds', fontsize=8)
    axs[0].legend(frameon=False, ncol=3, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=6.5)
    head = [best_val('off_n10_head42b_it400_s42')] + [best_val(f'off_n10_head42b_it400_s{s}') for s in (0, 1, 2)] + [None, None]
    exact = [best_val('off_n10_exact_s2_it280_s42')] + [best_val(f'off_final_n10_s{s}') for s in (0, 1, 2, 3, 4)]
    for k, (lab, vals, c) in enumerate([('seed-42 head', head, 'C4'), ('exact seed-42 run', exact, 'C3')]):
        xs = [i + (k - 0.5) * 0.36 for i, v in enumerate(vals) if v is not None]
        axs[1].bar(xs, [100 * v for v in vals if v is not None], 0.36, color=c, label=lab)
    axs[1].set_xticks(range(6)); axs[1].set_xticklabels([str(s) for s in seeds])
    axs[1].set_ylim(35, 85); axs[1].set_xlabel('seed of baseline.py')
    axs[1].set_title('$n = 10$, built for the seed-42 run', fontsize=8)
    axs[1].legend(frameon=False, ncol=2, loc='upper center', bbox_to_anchor=(0.5, -0.3), fontsize=6.5)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / 'per_seed.pdf'); plt.close(fig)


# ---------------------------------------------------------------- Fig. 5: images
def load(p):
    return np.asarray(Image.open(p).convert('RGB')).astype(np.float32)


def fig_images():
    med = json.load(open('paper/figdata/medoids_n10_files.json'))
    sub = Path('submissions/2026-10-15_n10_exact42_stage2')
    rows = [r.strip().split(',') for r in open(sub / 'train.csv')][1:]
    fig, axs = plt.subplots(3, 10, figsize=(TWO, 2.35))
    for c in range(10):
        a = load(Path('data/train') / med[c]['file'])
        b = load(sub / 'train' / rows[c][0])
        d = np.abs(b - a).mean(2)
        for r, img in enumerate([a / 255, b / 255]):
            axs[r, c].imshow(img)
        axs[2, c].imshow(d, cmap='magma', vmin=0, vmax=8)
        axs[0, c].set_title(NAMES[c], fontsize=6.5)
        for r in range(3):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
    for r, lab in enumerate(['medoid', 'distilled', '$|\\Delta|$ (0-8)']):
        axs[r, 0].set_ylabel(lab, fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(OUT / 'images_n10.pdf', dpi=300); plt.close(fig)
    # n = 120: one image per class, medoid vs scale-free PFD, difference x 40
    med = json.load(open('paper/figdata/medoids_n120_files.json'))
    sub = Path('submissions/2026-10-14_pfd3_n120_scalefree')
    rows = [r.strip().split(',') for r in open(sub / 'train.csv')][1:]
    fig, axs = plt.subplots(2, 10, figsize=(TWO, 1.55))
    for c in range(10):
        i = 12 * c
        a = load(Path('data/train') / med[i]['file']); b = load(sub / 'train' / rows[i][0])
        axs[0, c].imshow(b / 255)
        axs[1, c].imshow(np.clip(0.5 + 40 * (b - a) / 255, 0, 1))
        for r in range(2):
            axs[r, c].set_xticks([]); axs[r, c].set_yticks([])
        axs[0, c].set_title(NAMES[c], fontsize=6.5)
    axs[0, 0].set_ylabel('distilled', fontsize=7); axs[1, 0].set_ylabel('$40\\,\\Delta$', fontsize=7)
    fig.subplots_adjust(wspace=0.04, hspace=0.06)
    fig.savefig(OUT / 'images_n120.pdf', dpi=300); plt.close(fig)


# ---------------------------------------------------------------- Fig. 6: tuning to the seed-42 run
def fig_seed42():
    fig, axs = plt.subplots(1, 2, figsize=(TWO, 2.15))
    a = evals('pfd3_n10_head42.jsonl')
    it = [r['it'] for r in a]
    axs[0].plot(it, [100 * r['val_accs'][0] for r in a], 'o-', ms=2.5, color='C3', label='seed 42')
    axs[0].plot(it, [100 * r['val_accs'][1] for r in a], 's--', ms=2.5, color='0.45', label='seed 0')
    axs[0].plot(it, [100 * r['val_accs'][2] for r in a], 'v--', ms=2.5, color='0.65', label='seed 1')
    axs[0].set_title('fixed seed-42 head, random crops', fontsize=8)
    axs[0].set_xlabel('outer step'); axs[0].set_ylabel('accuracy (%)'); axs[0].set_ylim(54, 84)
    axs[0].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.36), ncol=3)
    e1, e2 = evals('pfd3_n10_exact.jsonl'), evals('pfd3_n10_exact_s2.jsonl')
    xs = [r['it'] for r in e1] + [200 + r['it'] for r in e2[1:]]
    axs[1].plot(xs, [100 * r['exact_acc'] for r in e1] + [100 * r['exact_acc'] for r in e2[1:]], 'o-', ms=2.5,
                color='C3', label='seed-42 run, recorded crops')
    axs[1].plot(xs, [100 * r['val_accs'][0] for r in e1] + [100 * r['val_accs'][0] for r in e2[1:]], 'o--', ms=2.5,
                color='C3', alpha=0.45, label='seed-42 head, random crops')
    axs[1].plot(xs, [100 * r['val_accs'][1] for r in e1] + [100 * r['val_accs'][1] for r in e2[1:]], 's--', ms=2.5,
                color='0.45', label='seed 0')
    m = evals('pfd3_n10_exact_med.jsonl')
    axs[1].plot([r['it'] for r in m], [100 * r['exact_acc'] for r in m], '-', color='C3', alpha=0.35, lw=0.9,
                label='recorded crops, from the medoids')
    axs[1].axvline(200, color='0.8', lw=0.6, zorder=0)
    axs[1].set_title('exact seed-42 run', fontsize=8)
    axs[1].set_xlabel('outer step'); axs[1].set_ylim(40, 84)
    axs[1].legend(frameon=False, loc='upper center', bbox_to_anchor=(0.5, -0.36), fontsize=6, ncol=2,
                  columnspacing=1.0)
    fig.tight_layout(w_pad=1.0)
    fig.savefig(OUT / 'seed42.pdf'); plt.close(fig)


# ---------------------------------------------------------------- Fig. 7: step size on the pixels
def fig_steps():
    rows = json.load(open(LOGS / 'diag_step_med120.json'))
    fig, ax = plt.subplots(figsize=(ONE, 2.1))
    style = {'lin': ('o-', 'C0', 'meta-gradient'), 'sign': ('s-', 'C1', 'sign of meta-gradient'),
             'rand_lin': ('x--', '0.5', 'same magnitudes, random signs'), 'rand_sign': ('v--', '0.7', 'random signs')}
    for k, (m, c, lab) in style.items():
        rs = sorted([r for r in rows if r['kind'] == k and r['quant']], key=lambda r: r['mean_dx_255'])
        ax.plot([r['mean_dx_255'] for r in rs], [100 * r['dAcc'] for r in rs], m, color=c, ms=3, label=lab)
    lp = [json.loads(l) for l in open(LOGS / 'diag_step_lp_med.out') if l.startswith('{"kind"')]
    lp = sorted([r for r in lp if r['kind'] == 'lp4'], key=lambda r: r['mean_dx_255'])
    ax.plot([r['mean_dx_255'] for r in lp], [100 * r['dAcc'] for r in lp], 'd:', color='C2', ms=3,
            label='meta-gradient, smoothed (4x4 blocks)')
    ax.axhline(0, color='0.3', lw=0.5)
    ax.set_xscale('log'); ax.set_ylim(-8, 2.5); ax.set_xlim(0.02, 4)
    ax.set_xlabel('mean pixel change (1/255 units)'); ax.set_ylabel('change in accuracy (points)')
    ax.legend(frameon=False, loc='upper center', bbox_to_anchor=(0.45, -0.22), ncol=2, fontsize=6, columnspacing=0.8)
    fig.savefig(OUT / 'step_size.pdf'); plt.close(fig)


if __name__ == '__main__':
    for f in [fig_score, fig_frontier, fig_curves120, fig_seeds, fig_images, fig_seed42, fig_steps]:
        try:
            f(); print('ok', f.__name__)
        except Exception as e:          # keep going: some logs may be missing while runs are in progress
            print('skip', f.__name__, type(e).__name__, e)
