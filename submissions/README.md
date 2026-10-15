# Submitted sets

Each folder is one Kaggle submission:

    train/                  the exact PNG files (64x64 RGB) we would hand to the organizers
    train.csv               file names and labels, in the order baseline.py reads them
    sha256.txt              sha256 of every PNG
    kaggle_submission.csv   the csv we uploaded (test predictions of baseline.py with its default arguments)
    scores.json             number of images, official validation accuracy (seed 42), public Kaggle score

To check one, see "Check a submission" in the main README. The row order in `train.csv` matters for the seed-42
sets, because the recorded crops are tied to row positions. `src/build_set.py` keeps it.

Below are the commands that made each set. They are the commands we ran, with paths made relative to the repo
root. Distillation runs are not seeded (see the main README), so running them again gives similar sets, not the
same pixels. "Best step" is the checkpoint we kept. For the multi-seed runs it is the best mean of the replica on
seeds 42, 0 and 1. For the seed-42 runs it is the best `exact_acc`.

## 2026-10-12_medoids_n120 (0.69171)

Real training images, not modified: 12 k-means medoids per class in ViT-B/16 feature space.

    python src/make_medoids.py 120                                  # cache/medoids_n120.pt = inits/medoids_n120.pt
    python src/export_set.py inits/medoids_n120.pt sets/medoids_n120

## 2026-10-13_pfd3_n120 (0.70258)

PFD from the medoids, cross-entropy outer loss, two random heads per step. Best step 250.

    python src/pfd3.py --init inits/medoids_n120.pt --window 20 --offload --heads 2 --iters 300 --tag n120
    python src/export_synth.py cache/pfd3_n120_best.pt sets/pfd3_n120

## 2026-10-14_pfd3_n120_scalefree (0.70344)

The previous set, then the scale-free outer loss (`--outer-T 0`): 100 steps (best step 100), then 200 more steps
with a smaller final step size (best step 200).

    python src/pfd3.py --init cache/pfd3_n120_best.pt --window 20 --offload --heads 2 --lr 0.03 --lr-end 0.01 \
        --iters 100 --outer-T 0 --tag n120_T0_fromopt
    python src/pfd3.py --init cache/pfd3_n120_T0_fromopt_best.pt --window 20 --offload --heads 2 --lr 0.03 \
        --lr-end 0.005 --iters 200 --outer-T 0 --tag n120_T0_stage2
    python src/export_synth.py cache/pfd3_n120_T0_stage2_best.pt sets/pfd3_n120_scalefree

Official validation on seeds 42, 0, 1, 2, 3, 4: 88.40, 90.69, 89.65, 91.16, 89.51, 90.22.

## 2026-10-15_n10_head42 (0.74258)

One image per class. The first two steps use our first PFD script (`pfd.py`, 5 training steps fully unrolled, random
heads). The third step keeps the classifier head that `baseline.py` draws with seed 42, with random crops. Best step
400 (kept with `--save-every-eval`).

    python src/pfd.py --init inits/medoids_n10.pt --opt sgdnorm --lr 0.05 --heads 2 --iters 400 --eval-every 50 --tag v2
    python src/pfd.py --init cache/pfd_v2_best.pt --opt sgdnorm --lr 0.03 --lr-end 0.005 --heads 4 --iters 400 \
        --eval-every 50 --tag v3
    python src/pfd3.py --init cache/pfd_v3_best.pt --offload --heads 2 --lr 0.05 --lr-end 0.01 --iters 400 \
        --eval-every 25 --outer-T 0 --fixed-head-seed 42 --save-every-eval --tag n10_head42b
    python src/export_synth.py cache/pfd3_n10_head42b_it400.pt sets/n10_head42

Official validation on seeds 42, 0, 1, 2: 71.60, 58.20, 58.74, 62.79.

## 2026-10-15_n10_exact42 (0.78135)

The previous set, then optimized on the exact seed-42 run: same head, and the batches, crops and flips recorded from
the official DataLoader. Best step 200.

    python src/record_views.py sets/n10_head42 42 cache/views_n10_seed42.pt     # = inits/views_n10_seed42.pt
    PROTO_PIL=1 python src/pfd3.py --init cache/pfd3_n10_head42b_it400.pt --offload --heads 1 --lr 0.03 \
        --lr-end 0.005 --iters 200 --eval-every 20 --outer-T 0 --fixed-head-seed 42 \
        --views inits/views_n10_seed42.pt --views-prob 1.0 --select-exact --fp32-inner --save-every-eval --tag n10_exact
    python src/export_synth.py cache/pfd3_n10_exact_it200.pt sets/n10_exact42

## 2026-10-15_n10_exact42_stage2 (0.78530)

300 more steps on the exact run with a smaller step size. Best step 280.

    PROTO_PIL=1 python src/pfd3.py --init cache/pfd3_n10_exact_it200.pt --offload --heads 1 --lr 0.02 \
        --lr-end 0.003 --iters 300 --eval-every 20 --outer-T 0 --fixed-head-seed 42 \
        --views inits/views_n10_seed42.pt --views-prob 1.0 --select-exact --fp32-inner --save-every-eval --tag n10_exact_s2
    python src/export_synth.py cache/pfd3_n10_exact_s2_it280.pt sets/n10_exact42_stage2

Official validation on seed 42: 80.62. On seeds 0 to 4: 54.81, 58.79, 58.35, 46.35, 42.89.
