# Interactional intelligence as an evolutionary channel — draft v0.5

This package contains the v0.5 manuscript draft, figures, data and code for the theory paper.

## Main changes in v0.5

1. Added a front conceptual figure that makes the coupled cognitive unit, channel geometry, branching design and measurement map visible before the equations.
2. Added a proposition-style core linking proposal variation, evaluative gradient, curvature and expected progress.
3. Moved the local NK channel index into the centre of the main result. The main NK figure now shows conditional strong complementarity, local channel prediction, held-out prediction gains and the channel closing along a trajectory.
4. Added a measurement table mapping symbols to empirical estimators.
5. Reworked the branching-conversation and agent sections into executable empirical designs.
6. Added initial robustness checks for the local-channel prediction under changes in stabilisation margin, state size and model proposal number.
7. Added Methods, Data and Code Availability, AI-use disclosure, Extended Data legends and a reproducibility roadmap.

## Key files

- `interactional_intelligence_nature_draft_v0_5.pdf` — compiled manuscript preview.
- `interactional_intelligence_nature_draft_v0_5.tex` — LaTeX source.
- `figures/fig1_conceptual_framework.pdf` — conceptual framework figure.
- `figures/fig2_nk_local_channel.pdf` — main NK local-channel figure.
- `extended_data/extended_data_fig1_nk_summary.pdf` — descriptive NK outcomes.
- `extended_data/extended_data_fig_robustness.pdf` — sensitivity checks.
- `code/` — simulation, local-channel analysis, robustness and figure-building scripts.
- `data/` — generated CSV outputs.

## Reproducing the current outputs

From the package root:

```bash
python code/run_nk_fast.py
python code/run_local_channel_fast.py
python code/nk_robustness_checks.py
python code/make_nature_figures.py
pdflatex -interaction=nonstopmode interactional_intelligence_nature_draft_v0_5.tex
pdflatex -interaction=nonstopmode interactional_intelligence_nature_draft_v0_5.tex
```

The scripts use fixed seeds. The NK simulation is illustrative rather than a psychological model of human or AI behaviour.

## Python requirements

- numpy
- pandas
- matplotlib

