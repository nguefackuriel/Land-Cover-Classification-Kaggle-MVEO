# Land cover classification from a handful of training images

This is our entry for the [MVEO 2026 Data-Centric Land Cover Classification Challenge](https://www.kaggle.com/competitions/data-centric-land-cover-classification-challenge-26)
(team "Nguefack Uriel"; Uriel Nguefack Yefou, Concordia University, uriel.nguefackyefou@mail.concordia.ca).
In this challenge the model and the training code are fixed. You only choose the training
images. The organizers' script (`baseline.py`) fine-tunes an ImageNet ViT-B/16 on your images for 5 epochs with AdamW
(lr 3e-5, batch 32), and the score rewards both test accuracy and a small number of images:

    score = 0.5 * accuracy + 0.5 * (1 - ln(n) / ln(18900))

The data is EuroSAT RGB (64x64 pixels, 10 classes). The repo has the code we wrote, the exact images we submitted,
and the commands to get our numbers again.

## Results

All validation numbers below come from the unmodified `baseline.py`. Kaggle scores are on the public leaderboard.

| Date | n | Training set | Val, seed 42 | Val, other seeds | Kaggle score |
|---|---|---|---|---|---|
| 12 Oct | 120 | k-means medoids (real images, untouched) | 86.49 | 87.19 ± 0.95 (6 seeds) | 0.69171 |
| 13 Oct | 120 | medoids refined with PFD (cross-entropy outer loss) | 88.37 | 89.27 ± 0.70 (6 seeds) | 0.70258 |
| 14 Oct | 120 | same, then the scale-free outer loss | 88.40 | 89.94 ± 0.98 (6 seeds) | 0.70344 |
| 15 Oct | 10 | PFD with the seed-42 classifier head fixed | 71.60 | 59.9 (seeds 0, 1, 2: 58.2, 58.7, 62.8) | 0.74258 |
| 15 Oct | 10 | PFD on the exact seed-42 training run | 79.38 | not measured | 0.78135 |
| 15 Oct | 10 | same, longer | 80.62 | 52.2 (seeds 0 to 4: 54.8, 58.8, 58.4, 46.4, 42.9) | 0.78530 |
| 19 Oct | 10 | sign matching toward a reachable network, 3 rounds, each followed by PFD | 83.88 | not measured | 0.80357 |
| 19 Oct | 10 | same, then sign matching and the PFD gradient together | 84.35 | not measured | 0.80567 |
| 20 Oct | 10 | same method on 224x224 images, two combined rounds | 84.74 | not measured | 0.80801 |
| 21 Oct | 10 | 224x224, from k-means medoids, three combined rounds, 4,096 train rows held out | 85.60 | not measured | 0.80826 |
| 21 Oct | 10 | same, five combined rounds | 86.07 | not measured | 0.81184 |
| 21 Oct | 10 | same, seven combined rounds | **86.44** | not measured | **0.81357** |

"Val, other seeds" means `baseline.py --seed s` for other values of `s`. For the 120-image sets it is the mean over
seeds 42 and 0 to 4. The last three sets are tuned to the run that `baseline.py` makes with its default settings, so
they only work for that run. Please read [About the seed-42 sets](#about-the-seed-42-sets) below.

## What is in the repo

    src/           all our code (scripts, run from the repo root)
    submissions/   the sets we submitted: the exact PNG files, train.csv, sha256 sums, the Kaggle csv and scores
    inits/         small starting files: k-means medoid indices and the recorded seed-42 training views
    results/       logs of every run we cite (results/logs) and the history.json of every official run
    paper/         the paper draft, its figures and the script that draws them
    checkpoints/   small saved image sets (.pt) and recorded views, so a run can restart from where it stopped
    notes/         our working notes (PLAN.md, in French), updated as we go
    tools/         sync_from_work.py, which copies our working folder into this repo

## Setup

We used one A100 80GB, Python 3.10, PyTorch 2.5.1 with CUDA 12.1.

1. Install the packages:

       pip install -r requirements.txt

2. Get the challenge data with the Kaggle CLI and unzip it into `data/`, so that `data/` holds `classes.json`,
   `train.csv`, `train/`, `validation.csv`, `validation/`, `test.csv` and `test/`:

       kaggle competitions download -c data-centric-land-cover-classification-challenge-26
       unzip data-centric-land-cover-classification-challenge-26.zip -d data

3. Get the official training script at the version we used:

       git clone https://github.com/mveo/challenge_2026_public ext/challenge_2026_public
       git -C ext/challenge_2026_public checkout b2c5d0045ed71a618d30dd475abd98f964195bb6

4. Make the working folders. Run every command in this README from the repo root.

       mkdir -p cache runs sets

The first script you run caches the data as tensors in `cache/` (a few minutes, about 1 GB).

## Check a submission (exact)

This rebuilds a challenge-format folder from the submitted PNGs, trains with the official script and compares the
test predictions with what we sent to Kaggle:

    python src/build_set.py submissions/2026-10-15_n10_exact42_stage2 sets/final
    python ext/challenge_2026_public/baseline.py train --data sets/final --output runs/final
    python src/compare_predictions.py runs/final/submission.csv submissions/2026-10-15_n10_exact42_stage2/kaggle_submission.csv

On our machine the last epoch prints `validation_accuracy` 0.8062 and the check prints
`4050 / 4050 test predictions identical`. The run is deterministic: we ran it three times with the same result. On
another GPU type or another PyTorch version, small float differences are possible. For the seed-42 sets we tested
this by switching TF32 on and off for matrix products and convolutions. The accuracy moved by less than 0.1 point.

The same three commands work for any folder in `submissions/`.

## Re-run the distillation

### The method in short

PFD ("protocol-faithful distillation") starts from real training images and changes their pixels so that the
official training run ends with a better model. It simulates the official training exactly (shuffling, batches of
32 with the last smaller batch kept, random resized crops and flips, AdamW, the per-epoch cosine schedule). It then
measures the loss of the trained network on 512 real training images and back-propagates through all the training
steps to the pixels. To keep memory flat, the backward pass rebuilds one training step at a time from saved states
(`--offload` keeps those states in CPU memory).

Three details matter a lot:

* Adam's first steps behave like sign descent, so the exact gradient through Adam is almost zero. In the backward
  pass we use a smoothed Adam step (`--tau`), and the forward pass stays exact.
* The new classifier head is random and barely moves in 20 small steps. Every outer step draws a new head, so the
  images work for any head. `--fixed-head-seed 42` instead uses the head that `baseline.py` draws with seed 42.
* The outer loss. Plain cross-entropy mostly rewards confident logits. With `--outer-T 0`, the loss is the
  cross-entropy at the temperature that fits the real batch best, so only the ranking of the classes counts. This
  gave +0.67 points on 6 official seeds at n = 120.

`src/pfd3.py` is the main script. It prints the replica accuracy on validation every `--eval-every` steps and keeps
the best images in `cache/pfd3_<tag>_best.pt`. With `--save-every-eval` it keeps every checkpoint as
`cache/pfd3_<tag>_it<step>.pt`. `src/export_synth.py` writes a checkpoint as a challenge folder of PNG files.
`src/official_seeds.py` then runs `baseline.py` on it for the seeds you give.

### n = 120 sets (good on every seed)

    # 1. k-means medoids in ViT feature space, 12 per class (same as inits/medoids_n120.pt)
    python src/make_medoids.py 120
    # 2. PFD with the cross-entropy outer loss, about 1 h 40 (best step 250: 2026-10-13 submission)
    python src/pfd3.py --init inits/medoids_n120.pt --window 20 --offload --heads 2 --iters 300 --tag n120
    # 3. scale-free outer loss, 100 then 200 more steps (2026-10-14 submission)
    python src/pfd3.py --init cache/pfd3_n120_best.pt --window 20 --offload --heads 2 --lr 0.03 --lr-end 0.01 \
        --iters 100 --outer-T 0 --tag n120_T0_fromopt
    python src/pfd3.py --init cache/pfd3_n120_T0_fromopt_best.pt --window 20 --offload --heads 2 --lr 0.03 \
        --lr-end 0.005 --iters 200 --outer-T 0 --tag n120_T0_stage2
    # 4. write the PNGs and score them with the official script on 6 seeds
    python src/export_synth.py cache/pfd3_n120_T0_stage2_best.pt sets/n120_scalefree
    python src/official_seeds.py sets/n120_scalefree n120_scalefree 42 0 1 2 3 4

At n = 120 the run needs 12.5 GB of GPU memory and about 21 GB of pinned CPU memory (20 saved training states).
One step takes about 20 s on an A100 with two heads.

### n = 10 sets (made for the default seed-42 run)

    # 1. record the exact batches, crops and flips of the default baseline.py run (seed 42, 2 workers).
    #    They only depend on n, the seed and the loader settings, not on the images. Same as inits/views_n10_seed42.pt.
    python src/build_set.py submissions/2026-10-15_n10_head42 sets/n10_head42
    python src/record_views.py sets/n10_head42 42 cache/views_n10_seed42.pt
    # 2. optimize on that exact run, starting from one medoid per class (about 1 h; we got 80.35 at step 400)
    PROTO_PIL=1 python src/pfd3.py --init inits/medoids_n10.pt --offload --heads 1 --lr 0.05 --lr-end 0.005 \
        --iters 400 --eval-every 20 --outer-T 0 --fixed-head-seed 42 --views inits/views_n10_seed42.pt \
        --select-exact --fp32-inner --save-every-eval --tag n10_exact_med

`PROTO_PIL=1` makes our GPU copy of the official pipeline round the resized images to 8 bits the way PIL does
(use it with `--fp32-inner`, as we did).
With it, the replica stays within 0.3 point of the official run at every epoch (`src/views_check.py`). The log line `exact_acc` is
that replica's validation accuracy. `val_accs` shows the same images trained with random crops on seeds 42, 0 and 1.

The submitted n = 10 sets came from a longer chain. The full list of commands for each submission is in
[submissions/README.md](submissions/README.md).

### Sign matching (the 19 Oct sets)

With 10 images the official run makes 5 AdamW steps. Each step moves a weight by at most its learning rate, so after
5 steps every weight stays within R = 9e-5 of its ImageNet value (R is the sum of the 5 learning rates). PFD stopped
near 80.6% on the seed-42 run. To see how far from the limit that is, `src/ceiling3.py` searches that box directly,
with no images at all: it moves the weights to lower the loss on training images, and clips them to the box. It finds
networks at 88 to 91% validation accuracy. So the protocol is not the limit. Our images were.

Two facts made the next step possible:

* If we move every weight of such a network to the edge of the box (exactly +R or -R, same direction), it keeps its
  accuracy (88.3% against 88.1%). Only the sign of each weight's move matters.
* Early Adam steps are sign steps. If the batch gradient has the opposite sign of a weight's target move at all 5
  steps, Adam walks that weight straight to the edge.

`src/signmatch.py` changes the pixels so that, at each official step, the sign of the batch gradient matches the
target. Since the path to the target is known in advance, each step is scored at its own point of that path with its
recorded crops and flips. There is no unrolled loop, so the gradient is smooth and reliable, unlike the PFD
meta-gradient near its optimum. The loss is a softplus of the sign mismatch, summed over all weights.

What worked best, in order:

1. Compute the target near where our images already end (`src/replay_state.py`, then `src/ceiling3.py
   --init-state`). A target far away (from the ImageNet weights) or too close (few search steps) works less well.
2. Sign matching with Adam on the pixels (80.6 -> 82.2), then PFD again (82.6). PFD is useless at its own optimum, but
   it helps again after sign matching.
3. Repeat with a stronger target (83.3, then 83.9 after PFD).
4. Both gradients together (`--pfd-w 3`): sign matching pulls toward the target, the real loss keeps what already
   works (84.35 official).

The 10 images reach about 85% sign agreement with these targets, and that seems to be the limit we hit now.
`src/loop_rounds2.sh` and `src/loop_hybrid.sh` run the rounds. `src/exact_eval.py` scores candidate sets on
validation plus 8,192 training images, which is less noisy than validation alone, and we use it to pick sets.
Those training images are not held out by default: the near target and the real-data gradients sample the whole
train split, so after a few rounds the train score goes up without test following (the third 224 round). With
`DD_HOLDOUT=k` (see `src/dd_lib.py`) the first k of those rows are never sampled by `ceiling3.py`, `signmatch.py` or
`pfd3.py`, and `exact_eval.py` scores them as `hold` instead.

### Randomness

The distillation runs are not seeded and use non-deterministic GPU kernels. Running them again gives images of the
same quality (in our experience within about 0.5 point), not the same pixels. That is why we keep the exact PNGs in
`submissions/`. Checking a submission is exact. Re-running the distillation is not.

## About the seed-42 sets

`baseline.py` has fixed defaults: `--seed 42` and `--workers 2`. With these defaults the whole training run is
fixed. The seed sets the classifier head (drawn on the CPU, so the same on every machine), the shuffling and the
crops. We recorded that run (`src/record_views.py` runs the official DataLoader itself and only records the random
draws) and tuned 10 images to it. On that run they reach 80.62% validation accuracy. With any other seed the same
images get 43 to 59% (mean 52.2%). The 120-image sets do not have this problem. They score about the same on every
seed.

So the seed-42 sets reproduce exactly if `baseline.py` runs with its default arguments, which is how we ran it and
how we understand the challenge protocol. They do not reproduce with another seed or another number of workers. On
19 Oct the organizers answered on the Kaggle forum: the final check runs `baseline.py` with its default settings,
seed 42 included, and a difference of a few tenths of a point is fine. They also said 64x64 PNG is preferred (not
required) and that images derived from the training set are fine. The repo keeps both kinds of sets.

## Scripts

| Script | What it does |
|---|---|
| `proto.py` | GPU copy of the official protocol (model, augmentations, evaluation), used inside all our loops |
| `dd_lib.py` | shared helpers: data cache, ViT features, k-means medoids, replica training, `load_set` |
| `pfd3.py` | the distillation (full unroll, checkpointed backward, all the options above) |
| `record_views.py` | records the exact batches, crops and flips of an official run |
| `build_set.py`, `export_synth.py`, `export_set.py`, `make_subset.py` | write challenge-format folders |
| `official_seeds.py`, `make_submission.py`, `compare_predictions.py` | run `baseline.py`, write and check Kaggle files |
| `make_medoids.py`, `make_medoids_km.py`, `select_eval.py`, `coreset_full_batches.py`, `coreset_big_n.py` | real-image coresets and the n sweep |
| `oracle_steps.py`, `ceiling_linf.py` | the full-data "oracle" and the best network reachable in a few steps |
| `views_check.py`, `views_robust.py`, `fidelity_check.py`, `fidelity_pil_eval.py` | replica against the official script |
| `diag_metagrad.py`, `diag_step.py`, `diag_tau.py` | checks of the meta-gradient and of the step size |
| `confusion.py`, `alloc_search.py`, `nsweep_eval.py`, `train_val_gap.py`, `basin_mix.py`, `delta_mix.py`, `mosaic_eval.py` | analyses and things that did not help |
| `pfd.py`, `pfd2.py`, `pfd_fo.py`, `make_x15.py` | earlier versions (`pfd.py` made the start of the n = 10 chain) |
| `ceiling3.py` | best network inside the 5-step box (fitted temperature, eval transform), from ImageNet or from a given state |
| `signmatch.py` | sign matching toward the edge of the box of a target network, optionally mixed with the PFD gradient (`--pfd-w`) |
| `replay_state.py`, `exact_eval.py`, `views_confusion.py` | replay the exact seed-42 run: final weights, val + 8k train accuracy, per-class results |
| `views_for_n.py` | records the seed-42 views for any n and any image size |
| `loop_rounds2.sh`, `loop_hybrid.sh` | the round pipelines (near target, sign matching, PFD) |
| `rough_exact.py`, `sparse_exact.py`, `tau_probe.py`, `corner_eval.py`, `agree_layers.py`, `cornerdesc.py`, `polish_sparse.py`, `subset_exact.py`, `upsample_init.py`, `ceiling2.py` | analyses and things that did not help |

## Things that did not help

These are in `results/logs` and in the paper:
* smooth (low-frequency) pixel changes
* 2x2 mosaics of real images
* more images for hard classes
* n = 97 (a one-image last batch)
* fp32 instead of bf16 inside the unrolled training
* 4 heads per step instead of 2
* mixing classes from two optimized sets
* fewer than 10 images on the seed-42 run (a class with no image is barely learned: n = 5 stayed under 47%)
* 224x224 images instead of 64x64 gave the same plateau (about 84%) with PFD and sign matching alone; only the hybrid round on 224 moved past the 64x64 hybrid (84.57 against 84.35, submitted)
* pulling the end of the training run toward a good network through the unrolled loop (`pfd3.py --target`)
* a sigmoid or weighted version of the sign matching loss, and "corner descent" (`cornerdesc.py`)
* small greedy pixel changes on the 8-bit grid (`polish_sparse.py`): within noise

At n = 120 every variant of the method ends near 90% (6 official seeds).
