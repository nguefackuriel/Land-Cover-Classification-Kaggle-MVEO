# Logs

`logs/` has the output of every run we cite, with the PyTorch warnings removed. `*.jsonl` files are the step-by-step
logs written by the distillation scripts. `*.out` files are what the scripts printed. `*.sh` files are the small
chains we used to run several jobs in a row. `official_runs/<name>_s<seed>/history.json` is the per-epoch validation
accuracy of each run of the unmodified `baseline.py` (`off_*`, `sub_*` and `recheck_*` runs).

Where to find the main numbers:

| Claim | Log |
|---|---|
| medoids at n = 120, 6 official seeds | `official_runs/sub_medoids_n120`, `official_runs/off_medoids_n120_s*` |
| PFD at n = 120 with the cross-entropy loss | `logs/pfd3_n120.out`, `official_runs/sub_pfd3_n120`, `official_runs/off_pfd3_n120_s*` |
| scale-free outer loss at n = 120 | `logs/pfd3_n120_T0_fromopt.out`, `logs/pfd3_n120_T0_stage2.out`, `logs/official_T0s2.out` |
| more heads, fp32/TF32, T = 0.1, a second basin, mixing classes | `logs/pfd3_n120_T0_stage3h4.out`, `logs/tf32_6seed.out`, `logs/official_T01.out`, `logs/km1_6seed.out`, `logs/mix_km0km1.out` |
| replica against the official script | `logs/fidelity_pil_eval.out`, `logs/views_check.out` |
| no overfitting of the training images | `logs/train_val_gap.out` |
| n between 104 and 120 | `logs/nsweep_medoids.out`, `logs/nsweep_km.out`, `logs/pfd3_n112_T0.out`, `logs/official_n112_T0.out` |
| step size and smooth steps | `logs/diag_step_*.out`, `logs/diag_tau_pfd120.out` |
| frontier n = 64, 96, 97 | `logs/pfd3_n64.out`, `logs/pfd3_n96.out`, `logs/pfd3_n97.out` |
| n = 10 with random heads (first PFD script) | `logs/pfd_v1.out`, `logs/pfd_v2.out`, `logs/pfd_v3.out` |
| n = 10 and 20 with the seed-42 head | `logs/pfd3_n10_head42.out`, `logs/pfd3_n10_head42b.out`, `logs/pfd3_n20_head42.out`, `logs/official_n10_head42b_it*.out` |
| n = 10 on the exact seed-42 run | `logs/pfd3_n10_exact*.out`, `logs/official_exact_it20.out`, `official_runs/off_n10_exact*` |
| final set on other seeds | `logs/official_final_n10_other_seeds.out` |
| float robustness of the seed-42 sets | `logs/views_robust.out` |

Accuracies in the logs are fractions (0.8062 = 80.62%). `val_accs` lists seeds 42, 0 and 1 of our GPU replica with
random crops. `exact_acc` is the replica on the recorded seed-42 run.
