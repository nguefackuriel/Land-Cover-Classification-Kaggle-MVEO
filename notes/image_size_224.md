# Why our main set uses 224x224 images

Team "Nguefack Uriel" (Uriel Nguefack Yefou, Concordia University). This note is for the post-challenge check. It
explains why our best set has 224x224 images, and what we give for the 64x64 case.

## What we hand over

| Set | Images | Official val (seed 42) | Kaggle score | Folder |
|---|---|---|---|---|
| Main | 10 PNG, 224x224 RGB, one per class | 88.47% | 0.82480 | `submissions/2026-11-04_n10_224_300it_rounds` |
| 64x64 | 10 PNG, 64x64 RGB, one per class | 85.04% | 0.81011 | `submissions/2026-11-04_n10_64_all_knobs_step_weighted` |

Both sets come from the same method and the same code. Each image is one synthetic image with one label. There
are no grids or tiles of several images. Both reproduce with the unmodified `baseline.py` and its default
arguments (seed 42, 2 workers). On our A100 two runs of each set gave the same predictions as the file we sent to
Kaggle. Switching TF32 on or off for matrix products and convolutions moves the result by at most 0.05 point in our
replica (224x224: 88.40 / 88.40 / 88.37 / 88.42; 64x64: 85.09 / 85.09 / 85.06 / 85.09), so another GPU should give
the same accuracy within the tolerance the organizers gave.

## Why 224x224

1. **The model works at 224x224.** `baseline.py` turns every training image into a 224x224 crop
   (`RandomResizedCrop(224, scale=(0.75, 1.0))`) and every test image into a 224x224 center crop. ViT-B/16 with
   IMAGENET1K_V1 weights was trained at that size: 14 x 14 patches of 16 x 16 pixels. A 64x64 image is upsampled
   about 3.5 times by a fixed bilinear filter before the model sees it. So a 64x64 file can set only 12,288 of the
   150,528 input values of each training view. The rest is fixed by the interpolation. A 224x224 file sets them all.
   We did not change the model, the training script or the evaluation. We only gave the model images at the size
   it already uses.

2. **The score counts images, not pixels.** The budget term is ln(n). Our set has n = 10 either way. In storage, the
   ten 224x224 PNGs take 0.94 MB, about 0.7% of the 128 MB EuroSAT training folder (the 64x64 set takes 0.09 MB). In
   raw values it is 1.5 million against 232 million for the training set, a 154x reduction.

3. **The extra pixels are used, and we measured it.** With the same ten starting images, the same recipe and the
   same held-out block, a 64x64 lineage and a 224x224 lineage were 0.7 point apart after the first round and 1.2
   points apart after five rounds. The 64x64 lineage stopped improving after its fourth round. Our final sets
   differ by 3.4 points on validation (88.47 vs 85.04) and 2.9 points on the Kaggle test (88.34 vs 85.41).
   The 224x224 sets are not 64x64 images in disguise: bringing the best 224x224 set down to 64x64 drops it from
   87.0% to 30-43% (box filter, bilinear filter, or a least-squares fit to the training views). The two sizes also
   lead the training run to different weights: the signs of the weight updates agree on only 62% of the weights.

4. **It is a finding about the task, not a trick.** Image resolution is a capacity knob for dataset distillation
   when the model input is fixed. With ten images and five AdamW steps, the 64x64 limit costs about a point at equal
   effort and three points at the end. Larger is not always better: 448x448 images were 0.4-0.5 point worse at every
   stage, because the training crop then has to shrink the image again. 224x224 is the size of the model input.

## Organizer statements we relied on (Kaggle forum)

* 19 Oct: 64x64 RGB PNG is preferred but not strict; images derived from the training set are fine; the check runs
  `baseline.py` with its default settings, and a few tenths of a point of difference is fine.
* 22 and 26 Oct: composite images (grids) are not accepted; 64x64 sets get priority in the checks unless larger
  images are justified. Our images are single images, and this note is our justification.

If the organizers prefer to rank 64x64 sets only, our 64x64 set above is the one to use.

All code, logs and the exact PNG files with their sha256 sums are in this repository (see the main README, section
"Check a submission").
