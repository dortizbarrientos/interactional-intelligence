from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nk_channel_simulation import run_grid, summarise, plot_final_by_condition, plot_phase_diagram, plot_trajectory, plot_bad_acceptance, write_manuscript_numbers

base = Path(__file__).resolve().parents[1]
data_dir = base/'data'; fig_dir = base/'figures'
data_dir.mkdir(exist_ok=True); fig_dir.mkdir(exist_ok=True)
# Moderate reps keep the illustrative run reproducible within a small package.
results, trajectories = run_grid(N=16, reps=32, steps=50, seed=20260611)
summary, deltas, params = summarise(results)
results.to_csv(data_dir/'nk_results.csv', index=False)
trajectories.to_csv(data_dir/'nk_trajectories_subset.csv', index=False)
summary.to_csv(data_dir/'nk_summary_by_condition.csv', index=False)
deltas.to_csv(data_dir/'nk_strong_complementarity.csv', index=False)
params.to_csv(data_dir/'nk_parameter_summary.csv', index=False)
plot_final_by_condition(summary, fig_dir, D=4)
plot_phase_diagram(deltas, fig_dir)
plot_trajectory(trajectories, fig_dir, K=4, D=4)
plot_bad_acceptance(summary, fig_dir, D=8)
write_manuscript_numbers(deltas, summary, base/'nk_manuscript_numbers.tex')
print(deltas[deltas.condition=='Strong_delta_HM_channel'].pivot(index='K', columns='D', values='mean_delta'))
