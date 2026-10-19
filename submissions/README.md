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

## 2026-10-19_n10_signmatch_r3 (0.80357)

Starts from the 0.78530 set. Rounds of: a target network near where the images end, sign matching toward the edge of
the box of that target, then PFD again. All on the exact seed-42 run. `V` is the recorded run.

    V=inits/views_n10_seed42.pt
    # round 1: near target (88.1%), sign matching (best step 120: 82.17), PFD (best step 140: 82.64)
    PROTO_PIL=1 python src/replay_state.py cache/pfd3_n10_exact_s2_it280.pt $V cache/state_n10_s2.pt
    python src/ceiling3.py --T 0 --opt adam --iters 300 --lr 0.02 --bs 256 --eval-every 50 --save-every 50 \
        --init-state cache/state_n10_s2.pt --tag near_s2
    PROTO_PIL=1 python src/signmatch.py --init cache/pfd3_n10_exact_s2_it280.pt --views $V \
        --target cache/ceiling3_near_s2.pt --iters 200 --opt adam --lr 1.0 --lr-end 0.1 --tag smA
    PROTO_PIL=1 python src/pfd3.py --init cache/sm_smA_best.pt --offload --heads 1 --lr 0.02 --lr-end 0.003 \
        --iters 150 --eval-every 20 --outer-T 0 --fixed-head-seed 42 --views $V --views-prob 1.0 --select-exact \
        --fp32-inner --no-seed-eval --save-every-eval --tag pf_smA
    # round 2 (no gain) is in results/logs (signmatch_r2*). Round 3: a stronger target (1000 search steps, 90.3%),
    # sign matching (best step 160: 83.31), PFD (best step 120: 83.88)
    PROTO_PIL=1 python src/replay_state.py cache/pfd3_pf_smA_best.pt $V cache/state_smA.pt
    python src/ceiling3.py --T 0 --opt adam --iters 1000 --lr 0.02 --bs 256 --eval-every 100 --save-every 250 \
        --init-state cache/state_smA.pt --tag near_smA_1k
    PROTO_PIL=1 python src/signmatch.py --init cache/pfd3_pf_smA_best.pt --views $V \
        --target cache/ceiling3_near_smA_1k_it500.pt --iters 200 --opt adam --lr 0.7 --lr-end 0.07 --warmup 20 --tag r3n
    PROTO_PIL=1 python src/pfd3.py --init cache/sm_r3n_best.pt --offload --heads 1 --lr 0.02 --lr-end 0.003 \
        --iters 120 --eval-every 20 --outer-T 0 --fixed-head-seed 42 --views $V --views-prob 1.0 --select-exact \
        --fp32-inner --no-seed-eval --tag pf_r3n
    python src/export_synth.py cache/pfd3_pf_r3n_best.pt sets/n10_sm_r3_pf

Official validation on seed 42: 83.88 (two runs, identical predictions). Kaggle test is about 84.10.

## 2026-10-19_n10_signmatch_hybrid (0.80567)

The previous set, then sign matching and the PFD gradient together. Both gradients are scaled to the same size and
the PFD part counts 3 times more. The target is searched from the end of the previous set's run, with the PIL-exact
evaluation. We picked step 120 with `src/exact_eval.py` (validation plus 8,192 training images): 84.99 against 84.28
for the previous set.

    PROTO_PIL=1 python src/replay_state.py cache/pfd3_pf_r3n_best.pt $V cache/state_pf_r3n.pt
    PROTO_PIL=1 python src/ceiling3.py --T 0 --opt adam --iters 1500 --lr 0.03 --bs 256 --eval-every 250 \
        --save-every 500 --init-state cache/state_pf_r3n.pt --tag fn_r3n
    PROTO_PIL=1 python src/signmatch.py --init cache/pfd3_pf_r3n_best.pt --views $V \
        --target cache/ceiling3_fn_r3n_it1000.pt --iters 150 --opt adam --lr 0.2 --lr-end 0.02 --warmup 10 \
        --pfd-w 3.0 --real-bs 1024 --tag hyb3
    python src/export_synth.py cache/sm_hyb3_it120.pt sets/n10_hyb3_it120

Official validation on seed 42: 84.35 (two runs, identical predictions). Kaggle test is about 84.52.
