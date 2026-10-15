# Paper

`main.tex` is the draft (IEEE journal format, `IEEEtran`), `refs.bib` the references, `figures/` the figures.

Every number in the text comes from a file in `results/`. The figures are drawn from those files by
`make_figures.py`. The image figures also read the original training images in `data/` (see the main README).

Redraw the figures (from the repo root):

    pip install matplotlib numpy pillow
    python paper/make_figures.py

Compile (any LaTeX distribution with `IEEEtran`; we used [tectonic](https://tectonic-typesetting.github.io)):

    cd paper && tectonic main.tex

`figdata/medoids_n10_files.json` and `figdata/medoids_n120_files.json` give the file names of the starting medoids
(rows of `inits/medoids_n10.pt` and `inits/medoids_n120.pt` in `data/train.csv`), so the figure script does not
need PyTorch.

Still to do: a final pass once the challenge is over (final leaderboard, answer of the organizers about the
evaluation seed).
