from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parent))
from nk_local_channel_index import (
    ChannelSamplingParameters, run_channel_grid, add_future_gains, prediction_table,
    bin_local_channel_index, plot_four_panel_nk_summary, plot_local_channel_results,
    write_channel_numbers
)

base = Path(__file__).resolve().parents[1]
data_dir = base/'data'; fig_dir = base/'figures'
data_dir.mkdir(exist_ok=True); fig_dir.mkdir(exist_ok=True)
channel_params = ChannelSamplingParameters(n_channel_samples=96, horizons=(1,3,5,10))
trajectories, local = run_channel_grid(N=16, reps=32, steps=50, seed=20260611, channel_params=channel_params)
local_future = add_future_gains(trajectories, local, channel_params.horizons)
pred = prediction_table(local_future, channel_params.horizons)
binned = bin_local_channel_index(local_future, horizon=5, n_bins=10)
trajectories.to_csv(data_dir/'nk_channel_trajectories.csv', index=False)
local.to_csv(data_dir/'nk_local_channel_index.csv', index=False)
local_future.to_csv(data_dir/'nk_local_channel_with_future_gains.csv', index=False)
pred.to_csv(data_dir/'nk_local_channel_prediction.csv', index=False)
binned.to_csv(data_dir/'nk_local_channel_index_binned_h5.csv', index=False)
plot_four_panel_nk_summary(base)
plot_local_channel_results(local_future, pred, binned, fig_dir)
write_channel_numbers(pred, binned, local_future, base/'nk_local_channel_numbers.tex')
print(pred)
