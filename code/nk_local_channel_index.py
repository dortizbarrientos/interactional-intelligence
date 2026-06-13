"""
Local channel index for NK trajectories.

This script extends the minimal NK exploration used in the
interactional-intelligence manuscript.  It asks whether a local estimate of
channel alignment, computed at each state of a stabilised human--AI trajectory,
predicts realised improvement over the next few search steps.

The model is deliberately modest.  It is not a psychological model of humans or
language models.  It is a controlled toy landscape where we can separate three
quantities that the theory treats as distinct:

    1. accessible variation: the proposal distribution over local moves;
    2. local additive signal: the quality change predicted from single-bit moves;
    3. curvature/noise liability: the deviation between multi-bit moves and their
       additive prediction, plus evaluator uncertainty.

For a current NK state x_t and a proposal mask U, define

    Delta_t(U) = F(x_t xor U) - F(x_t)                         true change
    L_t(U)     = sum_{i in U} [F(x_t xor i) - F(x_t)]          local additive prediction
    R_t(U)     = Delta_t(U) - L_t(U)                           nonlinear residual

Under the condition-specific proposal distribution pi_t(U), the bounded local
channel index is

    A_NK(t) = E[(max(L_t(U),0))^2]
              / { E[(max(L_t(U),0))^2] + E[R_t(U)^2] + E[sigma_t(U)^2] }.

A_NK(t) is high when the moves available to the coupled system have positive
local additive signal and low nonlinear/noise liability.  Because a reliability
index alone does not measure the amount of possible progress, we also define a
local channel potential

    P_NK(t) = E[max(L_t(U),0)] * A_NK(t).

The empirical test is whether P_NK(t) predicts realised improvement
F(x_{t+h}) - F(x_t) for horizons h = 1, 3, 5 and 10.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import math
import sys
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt

# Allow import when the script is run from anywhere.
THIS_DIR = Path(__file__).resolve().parent
if str(THIS_DIR) not in sys.path:
    sys.path.insert(0, str(THIS_DIR))

from nk_channel_simulation import (  # noqa: E402
    AgentParameters,
    NKLandscape,
    draw_human_distance,
    draw_mask,
    draw_model_distance,
    evaluation_sd,
    precompute_masks,
)


# -----------------------------
# Figure style
# -----------------------------


def set_nature_like_style() -> None:
    """A compact vector-figure style close to Nature's production guidance."""
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "legend.fontsize": 6,
            "axes.linewidth": 0.6,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.major.size": 2.5,
            "ytick.major.size": 2.5,
            "lines.linewidth": 0.9,
            "lines.markersize": 3.0,
            "errorbar.capsize": 2,
            "figure.dpi": 300,
            "savefig.dpi": 450,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.015,
        }
    )


# -----------------------------
# Proposal distribution
# -----------------------------


@dataclass(frozen=True)
class ChannelSamplingParameters:
    n_channel_samples: int = 160
    horizons: Tuple[int, ...] = (1, 3, 5, 10)
    eps: float = 1.0e-12


def build_cdf_cache(D_values: Iterable[int]) -> Dict[int, np.ndarray]:
    """Distance CDF used by the model-like proposal distribution."""
    cdf_cache: Dict[int, np.ndarray] = {}
    for D in D_values:
        weights = np.arange(1, D + 1, dtype=float)
        weights /= weights.sum()
        cdf_cache[int(D)] = np.cumsum(weights)
    return cdf_cache


def draw_condition_mask(
    condition: str,
    D: int,
    masks: Dict[int, np.ndarray],
    cdf_cache: Dict[int, np.ndarray],
    rng: np.random.Generator,
    params: AgentParameters,
) -> Tuple[int, int, str]:
    """Draw one proposal mask from the condition-specific proposal distribution."""
    if condition == "H":
        source = "human"
    elif condition == "M":
        source = "model"
    elif condition in {"HM_naive", "HM_channel"}:
        p_human = params.n_human_proposals / (params.n_human_proposals + params.n_model_proposals)
        source = "human" if rng.random() < p_human else "model"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    if source == "human":
        distance = draw_human_distance(rng)
    else:
        distance = draw_model_distance(rng, D, cdf_cache)

    return draw_mask(rng, masks, distance), distance, source


def draw_human_distances_vec(rng: np.random.Generator, n: int) -> np.ndarray:
    """Vectorised version of the human-like proposal distance distribution."""
    r = rng.random(n)
    out = np.ones(n, dtype=np.int16)
    out[r >= 0.85] = 2
    out[r >= 0.97] = 3
    return out


def draw_model_distances_vec(rng: np.random.Generator, n: int, D: int, cdf_cache: Dict[int, np.ndarray]) -> np.ndarray:
    """Vectorised version of the model-like proposal distance distribution."""
    if D <= 1:
        return np.ones(n, dtype=np.int16)
    cdf = cdf_cache[D]
    return np.searchsorted(cdf, rng.random(n), side="right").astype(np.int16) + 1


def draw_condition_masks_vec(
    condition: str,
    D: int,
    n: int,
    masks: Dict[int, np.ndarray],
    cdf_cache: Dict[int, np.ndarray],
    rng: np.random.Generator,
    params: AgentParameters,
) -> Tuple[np.ndarray, np.ndarray]:
    """Draw proposal masks and distances for local channel estimation."""
    if condition == "H":
        source_is_human = np.ones(n, dtype=bool)
    elif condition == "M":
        source_is_human = np.zeros(n, dtype=bool)
    elif condition in {"HM_naive", "HM_channel"}:
        p_human = params.n_human_proposals / (params.n_human_proposals + params.n_model_proposals)
        source_is_human = rng.random(n) < p_human
    else:
        raise ValueError(f"Unknown condition: {condition}")

    distances = np.empty(n, dtype=np.int16)
    n_h = int(source_is_human.sum())
    n_m = n - n_h
    if n_h:
        distances[source_is_human] = draw_human_distances_vec(rng, n_h)
    if n_m:
        distances[~source_is_human] = draw_model_distances_vec(rng, n_m, D, cdf_cache)

    proposal_masks = np.empty(n, dtype=np.uint32)
    for d in np.unique(distances):
        idx = np.flatnonzero(distances == d)
        arr = masks[int(d)]
        proposal_masks[idx] = arr[rng.integers(0, len(arr), size=len(idx))]
    return proposal_masks, distances


def mask_bit_indices(mask: int, N: int) -> List[int]:
    """Return bit indices flipped by an integer mask."""
    return [i for i in range(N) if (mask >> i) & 1]


# -----------------------------
# Local channel estimator
# -----------------------------


def estimate_local_channel(
    landscape: NKLandscape,
    state: int,
    condition: str,
    D: int,
    masks: Dict[int, np.ndarray],
    cdf_cache: Dict[int, np.ndarray],
    rng: np.random.Generator,
    params: AgentParameters,
    channel_params: ChannelSamplingParameters,
) -> Dict[str, float]:
    """Estimate the local NK channel index at a trajectory state.

    The estimate uses sampled proposals from the condition's proposal
    distribution.  This deliberately mirrors what an empirical branching design
    would do: sample possible next moves around the same artefact, score them,
    and ask whether accessible variation is aligned with improvement.
    """
    N = landscape.N
    base = landscape.fitness(state)

    # Single-bit effects give the local additive approximation.
    bit_masks = (np.uint32(1) << np.arange(N, dtype=np.uint32))
    single_delta = landscape.fitness_array[np.bitwise_xor(np.uint32(state), bit_masks)] - base

    proposal_masks, distances = draw_condition_masks_vec(
        condition=condition,
        D=D,
        n=channel_params.n_channel_samples,
        masks=masks,
        cdf_cache=cdf_cache,
        rng=rng,
        params=params,
    )

    # Vectorised local additive prediction for each sampled multi-bit proposal.
    bit_matrix = ((proposal_masks[:, None] >> np.arange(N, dtype=np.uint32)) & np.uint32(1)).astype(float)
    linear = bit_matrix @ single_delta
    true_delta = landscape.fitness_array[np.bitwise_xor(np.uint32(state), proposal_masks)] - base
    residual = true_delta - linear

    evaluator = "model" if condition == "M" else "human"
    ruggedness = landscape.K / max(1, landscape.N - 1)
    if evaluator == "human":
        sd = 0.010 * (1.0 + 2.8 * ruggedness * distances.astype(float))
    else:
        sd = 0.035 * (1.0 + 1.8 * ruggedness * distances.astype(float))
    eval_var = sd * sd

    positive_linear = np.maximum(linear, 0.0)
    linear_signal_sq = float(np.mean(positive_linear**2))
    linear_signal = float(np.mean(positive_linear))
    curvature_liability = float(np.mean(residual**2))
    noise_liability = float(np.mean(eval_var))

    channel_index = linear_signal_sq / (
        linear_signal_sq + curvature_liability + noise_liability + channel_params.eps
    )
    channel_potential = linear_signal * channel_index

    misleading = (linear > 0.0) & (true_delta < 0.0)

    return {
        "quality": base,
        "local_linear_signal": linear_signal,
        "local_linear_signal_sq": linear_signal_sq,
        "local_curvature_liability": curvature_liability,
        "local_noise_liability": noise_liability,
        "local_channel_index": channel_index,
        "local_channel_potential": channel_potential,
        "local_true_gain_available": float(np.mean(np.maximum(true_delta, 0.0))),
        "local_best_true_gain": float(np.max(true_delta)),
        "local_misleading_fraction": float(np.mean(misleading)),
        "local_mean_proposal_distance": float(np.mean(distances)),
    }


# -----------------------------
# Trajectory simulation with channel logging
# -----------------------------


def proposal_batch_for_condition(
    condition: str,
    state: int,
    D: int,
    masks: Dict[int, np.ndarray],
    cdf_cache: Dict[int, np.ndarray],
    rng: np.random.Generator,
    params: AgentParameters,
) -> List[Tuple[int, int, str]]:
    """Generate the actual candidate proposals used by one search step."""
    proposals: List[Tuple[int, int, str]] = []

    if condition == "H":
        n_h, n_m = params.n_human_proposals, 0
        evaluator = "human"
    elif condition == "M":
        n_h, n_m = 0, params.n_model_proposals
        evaluator = "model"
    elif condition in {"HM_naive", "HM_channel"}:
        n_h, n_m = params.n_human_proposals, params.n_model_proposals
        evaluator = "human"
    else:
        raise ValueError(f"Unknown condition: {condition}")

    for _ in range(n_h):
        d = draw_human_distance(rng)
        proposals.append((state ^ draw_mask(rng, masks, d), d, evaluator))
    for _ in range(n_m):
        d = draw_model_distance(rng, D, cdf_cache)
        proposals.append((state ^ draw_mask(rng, masks, d), d, evaluator))
    return proposals


def simulate_condition_with_channel_logging(
    landscape: NKLandscape,
    start: int,
    condition: str,
    steps: int,
    D: int,
    masks: Dict[int, np.ndarray],
    cdf_cache: Dict[int, np.ndarray],
    rng: np.random.Generator,
    channel_rng: np.random.Generator,
    params: AgentParameters,
    channel_params: ChannelSamplingParameters,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run a trajectory and log local channel estimates at each state."""
    state = int(start)
    fitness = landscape.fitness(state)
    trajectory_rows: List[Dict[str, float]] = []
    local_rows: List[Dict[str, float]] = []

    for step in range(steps + 1):
        trajectory_rows.append(
            {
                "step": step,
                "state": state,
                "fitness": fitness,
            }
        )

        if step == steps:
            break

        local = estimate_local_channel(
            landscape=landscape,
            state=state,
            condition=condition,
            D=D,
            masks=masks,
            cdf_cache=cdf_cache,
            rng=channel_rng,
            params=params,
            channel_params=channel_params,
        )
        local_rows.append({"step": step, "state": state, **local})

        proposals = proposal_batch_for_condition(
            condition=condition,
            state=state,
            D=D,
            masks=masks,
            cdf_cache=cdf_cache,
            rng=rng,
            params=params,
        )

        if condition == "M":
            evaluator = "model"
            bias_strength = params.model_bias_strength
            margin_lambda = 0.0
        elif condition == "HM_channel":
            evaluator = "human"
            bias_strength = 0.0
            margin_lambda = params.channel_margin_lambda
        else:
            evaluator = "human"
            bias_strength = 0.0
            margin_lambda = 0.0

        best_estimate = -math.inf
        best: Tuple[int, float, int, float, float, float] | None = None

        for new_state, distance, _eval in proposals:
            new_fitness = landscape.fitness(new_state)
            true_step_delta = new_fitness - fitness
            sd = evaluation_sd(evaluator, landscape.K, landscape.N, distance)
            bias = bias_strength * (distance - 1) * (1.0 + landscape.K / max(1, landscape.N - 1))
            perceived_delta = true_step_delta + rng.normal(0.0, sd) + bias
            if perceived_delta > best_estimate:
                best_estimate = perceived_delta
                best = (new_state, new_fitness, distance, true_step_delta, sd, perceived_delta)

        assert best is not None
        new_state, new_fitness, distance, true_step_delta, sd, perceived_delta = best
        if perceived_delta > margin_lambda * sd:
            state = int(new_state)
            fitness = float(new_fitness)

    return pd.DataFrame(trajectory_rows), pd.DataFrame(local_rows)


def run_channel_grid(
    N: int = 16,
    K_values: Iterable[int] = (0, 2, 4, 8, 12),
    D_values: Iterable[int] = (1, 2, 4, 8),
    reps: int = 80,
    steps: int = 50,
    seed: int = 20260611,
    condition: str = "HM_channel",
    params: AgentParameters = AgentParameters(),
    channel_params: ChannelSamplingParameters = ChannelSamplingParameters(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run channel-index logging over the same paired NK grid as v0.3."""
    K_values = list(K_values)
    D_values = list(D_values)
    max_D = max(max(D_values), 3)
    masks = precompute_masks(N, max_D)
    cdf_cache = build_cdf_cache(D_values)

    condition_seed_offsets = {"H": 101, "M": 202, "HM_naive": 303, "HM_channel": 404}
    rng_main = np.random.default_rng(seed)

    all_traj: List[pd.DataFrame] = []
    all_local: List[pd.DataFrame] = []

    for K in K_values:
        for rep in range(reps):
            landscape_seed = int(rng_main.integers(1, 2_000_000_000))
            landscape_rng = np.random.default_rng(landscape_seed)
            landscape = NKLandscape(N, K, landscape_rng)
            start = int(landscape_rng.integers(0, 1 << N, dtype=np.uint32))
            start_fitness = landscape.fitness(start)

            for D in D_values:
                agent_seed = landscape_seed + condition_seed_offsets[condition] + 1009 * D
                agent_rng = np.random.default_rng(agent_seed)
                channel_rng = np.random.default_rng(agent_seed + 9_999_991)
                traj, local = simulate_condition_with_channel_logging(
                    landscape=landscape,
                    start=start,
                    condition=condition,
                    steps=steps,
                    D=D,
                    masks=masks,
                    cdf_cache=cdf_cache,
                    rng=agent_rng,
                    channel_rng=channel_rng,
                    params=params,
                    channel_params=channel_params,
                )
                for df in (traj, local):
                    df["N"] = N
                    df["K"] = K
                    df["D"] = D
                    df["rep"] = rep
                    df["condition"] = condition
                    df["landscape_seed"] = landscape_seed
                    df["start_fitness"] = start_fitness
                all_traj.append(traj)
                all_local.append(local)

    return pd.concat(all_traj, ignore_index=True), pd.concat(all_local, ignore_index=True)


# -----------------------------
# Prediction analysis
# -----------------------------


def add_future_gains(
    trajectories: pd.DataFrame,
    local: pd.DataFrame,
    horizons: Iterable[int],
) -> pd.DataFrame:
    """Add future improvement F(t+h)-F(t) to local-channel rows."""
    key_cols = ["N", "K", "D", "rep", "condition", "landscape_seed"]
    traj_keyed = trajectories[key_cols + ["step", "fitness"]].copy()
    out = local.copy()
    for h in horizons:
        future = traj_keyed.copy()
        future["step"] = future["step"] - h
        future = future.rename(columns={"fitness": f"future_fitness_h{h}"})
        out = out.merge(future, on=key_cols + ["step"], how="left")
        out[f"future_gain_h{h}"] = out[f"future_fitness_h{h}"] - out["quality"]
    return out


def standardise_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
    features: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return train/test design matrices with train-set centring and scaling."""
    train_X_raw = train[features].to_numpy(dtype=float)
    test_X_raw = test[features].to_numpy(dtype=float)
    mu = train_X_raw.mean(axis=0)
    sd = train_X_raw.std(axis=0, ddof=0)
    sd[sd == 0.0] = 1.0
    train_X = (train_X_raw - mu) / sd
    test_X = (test_X_raw - mu) / sd
    train_X = np.column_stack([np.ones(len(train_X)), train_X])
    test_X = np.column_stack([np.ones(len(test_X)), test_X])
    return train_X, test_X, mu, sd


def fit_ols_predict(train: pd.DataFrame, test: pd.DataFrame, features: List[str], target: str) -> Dict[str, object]:
    """Fit OLS by least squares and report out-of-sample R^2."""
    train = train.dropna(subset=features + [target]).copy()
    test = test.dropna(subset=features + [target]).copy()
    X_train, X_test, _mu, _sd = standardise_train_test(train, test, features)
    y_train = train[target].to_numpy(dtype=float)
    y_test = test[target].to_numpy(dtype=float)
    coef, *_ = np.linalg.lstsq(X_train, y_train, rcond=None)
    pred = X_test @ coef
    ss_res = float(np.sum((y_test - pred) ** 2))
    ss_tot = float(np.sum((y_test - y_test.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan
    corr = float(np.corrcoef(y_test, pred)[0, 1]) if len(y_test) > 2 else np.nan
    return {"r2": r2, "corr": corr, "coef": coef, "n_train": len(train), "n_test": len(test)}


def assign_group_split(df: pd.DataFrame, seed: int = 7071, test_fraction: float = 0.3) -> pd.Series:
    """Split by paired landscape, not by row, to avoid within-trajectory leakage."""
    groups = df[["K", "D", "rep", "landscape_seed"]].drop_duplicates().copy()
    rng = np.random.default_rng(seed)
    groups["is_test"] = rng.random(len(groups)) < test_fraction
    merged = df.merge(groups, on=["K", "D", "rep", "landscape_seed"], how="left")
    return merged["is_test"].astype(bool)


def prediction_table(local_with_future: pd.DataFrame, horizons: Iterable[int]) -> pd.DataFrame:
    """Compare baseline, index-only and fuller channel prediction of future gain."""
    rows: List[Dict[str, object]] = []
    df = local_with_future.copy()
    df["K_scaled"] = df["K"].astype(float)
    df["D_scaled"] = df["D"].astype(float)
    df["step_scaled"] = df["step"].astype(float)

    is_test = assign_group_split(df)
    train = df[~is_test].copy()
    test = df[is_test].copy()

    baseline_features = ["quality", "K_scaled", "D_scaled", "step_scaled"]
    index_features = baseline_features + ["local_channel_index"]
    channel_features = baseline_features + [
        "local_channel_index",
        "local_channel_potential",
        "local_misleading_fraction",
        "local_curvature_liability",
    ]

    for h in horizons:
        target = f"future_gain_h{h}"
        base_fit = fit_ols_predict(train, test, baseline_features, target)
        idx_fit = fit_ols_predict(train, test, index_features, target)
        chan_fit = fit_ols_predict(train, test, channel_features, target)
        rows.append(
            {
                "horizon": h,
                "model": "baseline",
                "r2_test": base_fit["r2"],
                "corr_test": base_fit["corr"],
                "n_train": base_fit["n_train"],
                "n_test": base_fit["n_test"],
                "coef_channel_index": np.nan,
                "coef_channel_potential": np.nan,
            }
        )
        idx_coef_index = index_features.index("local_channel_index") + 1
        rows.append(
            {
                "horizon": h,
                "model": "baseline_plus_index",
                "r2_test": idx_fit["r2"],
                "corr_test": idx_fit["corr"],
                "n_train": idx_fit["n_train"],
                "n_test": idx_fit["n_test"],
                "coef_channel_index": float(idx_fit["coef"][idx_coef_index]),
                "coef_channel_potential": np.nan,
            }
        )
        # Coefficients are on standardised predictors; feature order follows channel_features.
        channel_index_index = channel_features.index("local_channel_index") + 1
        channel_potential_index = channel_features.index("local_channel_potential") + 1
        rows.append(
            {
                "horizon": h,
                "model": "baseline_plus_channel",
                "r2_test": chan_fit["r2"],
                "corr_test": chan_fit["corr"],
                "n_train": chan_fit["n_train"],
                "n_test": chan_fit["n_test"],
                "coef_channel_index": float(chan_fit["coef"][channel_index_index]),
                "coef_channel_potential": float(chan_fit["coef"][channel_potential_index]),
            }
        )
    return pd.DataFrame(rows)


def bin_local_channel_index(df: pd.DataFrame, horizon: int, n_bins: int = 10) -> pd.DataFrame:
    """Bin the bounded local channel index and summarise future gains."""
    target = f"future_gain_h{horizon}"
    d = df.dropna(subset=["local_channel_index", target]).copy()
    # qcut can fail with repeated values, so rank first while preserving order.
    d["rank_for_bins"] = d["local_channel_index"].rank(method="first")
    d["bin"] = pd.qcut(d["rank_for_bins"], q=n_bins, labels=False)
    out = (
        d.groupby("bin", as_index=False)
        .agg(
            mean_channel_index=("local_channel_index", "mean"),
            mean_channel_potential=("local_channel_potential", "mean"),
            mean_future_gain=(target, "mean"),
            sem_future_gain=(target, lambda x: float(x.std(ddof=1) / math.sqrt(len(x)))),
            mean_misleading_fraction=("local_misleading_fraction", "mean"),
            n=(target, "size"),
        )
    )
    out["decile"] = out["bin"] + 1
    return out



# -----------------------------
# Plotting
# -----------------------------


def plot_four_panel_nk_summary(base: Path) -> None:
    """Regenerate the four existing NK panels as one editable vector figure."""
    set_nature_like_style()
    data_dir = base / "data"
    fig_dir = base / "figures"
    summary = pd.read_csv(data_dir / "nk_summary_by_condition.csv")
    deltas = pd.read_csv(data_dir / "nk_strong_complementarity.csv")
    trajectories = pd.read_csv(data_dir / "nk_trajectories_subset.csv")

    colours = {"H": "#1f77b4", "M": "#ff7f0e", "HM_naive": "#2ca02c", "HM_channel": "#d62728"}
    # 183 mm x 128 mm: Nature double-column width, below maximum figure height.
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.05))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    subset = summary[summary["D"] == 4]
    for condition in ["H", "M", "HM_naive", "HM_channel"]:
        s = subset[subset["condition"] == condition].sort_values("K")
        ax_a.errorbar(
            s["K"],
            s["mean_final"],
            yerr=s["sem_final"],
            marker="o",
            linewidth=0.9,
            capsize=2,
            label=condition,
            color=colours[condition],
        )
    ax_a.set_xlabel("Landscape ruggedness, K")
    ax_a.set_ylabel("Final quality")
    ax_a.legend(frameon=False, loc="lower left", handlelength=1.4)
    ax_a.set_xticks([0, 2, 4, 8, 12])

    subset_traj = trajectories[(trajectories["K"] == 4) & (trajectories["D"] == 4)]
    mean_traj = subset_traj.groupby(["condition", "step"], as_index=False).agg(mean_fitness=("fitness", "mean"))
    for condition in ["H", "M", "HM_naive", "HM_channel"]:
        s = mean_traj[mean_traj["condition"] == condition].sort_values("step")
        ax_b.plot(s["step"], s["mean_fitness"], linewidth=0.9, label=condition, color=colours[condition])
    ax_b.set_xlabel("Step")
    ax_b.set_ylabel("Quality")
    ax_b.legend(frameon=False, loc="lower right", handlelength=1.4)

    delta_subset = deltas[deltas["condition"] == "Strong_delta_HM_channel"]
    table = delta_subset.pivot(index="K", columns="D", values="mean_delta").sort_index()
    vmax = float(np.nanmax(np.abs(table.values)))
    im = ax_c.imshow(table.values, aspect="auto", origin="lower", cmap="viridis", vmin=-vmax, vmax=vmax)
    ax_c.set_xticks(np.arange(table.shape[1]), labels=[str(c) for c in table.columns])
    ax_c.set_yticks(np.arange(table.shape[0]), labels=[str(i) for i in table.index])
    ax_c.set_xlabel("Model proposal breadth, D")
    ax_c.set_ylabel("Landscape ruggedness, K")
    for i, K in enumerate(table.index):
        for j, D in enumerate(table.columns):
            val = table.loc[K, D]
            text_colour = "white" if val < -0.045 else "black"
            ax_c.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5.3, color=text_colour)
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.02)
    cbar.set_label(r"Mean $\Delta_{strong}$")
    cbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)

    subset_bad = summary[(summary["D"] == 8) & (summary["condition"].isin(["HM_naive", "HM_channel"]))]
    bad_colours = {"HM_naive": "#1f77b4", "HM_channel": "#ff7f0e"}
    for condition in ["HM_naive", "HM_channel"]:
        s = subset_bad[subset_bad["condition"] == condition].sort_values("K")
        ax_d.plot(s["K"], s["mean_bad_acceptance"], marker="o", linewidth=0.9, label=condition, color=bad_colours[condition])
    ax_d.set_xlabel("Landscape ruggedness, K")
    ax_d.set_ylabel("Accepted harmful moves")
    ax_d.set_xticks([0, 2, 4, 8, 12])
    ax_d.legend(frameon=False, loc="upper left", handlelength=1.4)

    panel_titles = [
        "Final quality (D = 4)",
        "Trajectories (K = 4, D = 4)",
        "Strong complementarity",
        "Harmful accepted moves (D = 8)",
    ]
    for label, ax, title in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d], panel_titles):
        ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top", ha="left")
        ax.set_title(title, loc="left", pad=2)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
        ax.grid(False)

    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    fig.savefig(fig_dir / "fig_nk_summary_4panel.pdf")
    fig.savefig(fig_dir / "fig_nk_summary_4panel.png", dpi=450)
    plt.close(fig)


def plot_local_channel_results(
    local_future: pd.DataFrame,
    pred: pd.DataFrame,
    binned: pd.DataFrame,
    outdir: Path,
) -> None:
    """Four-panel figure for the local channel index analysis."""
    set_nature_like_style()
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.05))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # Panel a: binned relationship between local channel potential and 5-step gain.
    ax_a.errorbar(
        binned["decile"],
        binned["mean_future_gain"],
        yerr=binned["sem_future_gain"],
        marker="o",
        linewidth=0.9,
        color="black",
    )
    ax_a.set_xlabel(r"Local channel index decile, $A^{NK}_t$")
    ax_a.set_ylabel(r"Realised gain, $F_{t+5}-F_t$")
    ax_a.set_xticks([1, 5, 10])

    # Panel b: out-of-sample R2 improvement across horizons.
    wide = pred.pivot(index="horizon", columns="model", values="r2_test").reset_index()
    ax_b.plot(wide["horizon"], wide["baseline"], marker="o", label="baseline", color="#7f7f7f")
    ax_b.plot(wide["horizon"], wide["baseline_plus_index"], marker="o", label="+ index", color="#1f77b4")
    ax_b.plot(
        wide["horizon"],
        wide["baseline_plus_channel"],
        marker="o",
        label="+ channel terms",
        color="black",
    )
    ax_b.set_xlabel("Prediction horizon, h")
    ax_b.set_ylabel(r"Test $R^2$")
    ax_b.set_xticks(list(wide["horizon"]))
    ax_b.legend(frameon=False, loc="best", handlelength=1.4)

    # Panel c: local channel potential across K and D.
    heat = local_future.groupby(["K", "D"], as_index=False).agg(mean_a=("local_channel_index", "mean"))
    table = heat.pivot(index="K", columns="D", values="mean_a").sort_index()
    im = ax_c.imshow(table.values, origin="lower", aspect="auto", cmap="viridis")
    ax_c.set_xticks(np.arange(table.shape[1]), labels=[str(c) for c in table.columns])
    ax_c.set_yticks(np.arange(table.shape[0]), labels=[str(i) for i in table.index])
    ax_c.set_xlabel("Model proposal breadth, D")
    ax_c.set_ylabel("Landscape ruggedness, K")
    for i, K in enumerate(table.index):
        for j, D in enumerate(table.columns):
            ax_c.text(j, i, f"{table.loc[K,D]:.3f}", ha="center", va="center", fontsize=5.3, color="black")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.02)
    cbar.set_label(r"Mean $A^{NK}_t$")
    cbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)

    # Panel d: example trajectory of quality and channel potential.
    subset = local_future[(local_future["K"] == 4) & (local_future["D"] == 4)]
    mean_by_step = subset.groupby("step", as_index=False).agg(
        mean_quality=("quality", "mean"),
        mean_channel=("local_channel_index", "mean"),
    )
    ax_d.plot(mean_by_step["step"], mean_by_step["mean_quality"], color="black", label="quality")
    ax_d.set_xlabel("Step")
    ax_d.set_ylabel("Quality")
    ax_d2 = ax_d.twinx()
    ax_d2.plot(mean_by_step["step"], mean_by_step["mean_channel"], color="#1f77b4", label=r"$A^{NK}_t$")
    ax_d2.set_ylabel(r"Local channel index")
    lines, labels = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines + lines2, labels + labels2, frameon=False, loc="center right", handlelength=1.4)

    panel_titles = [
        "Local channel index predicts gain",
        "Prediction improves across horizons",
        "Channel index over NK settings",
        "Channel index along trajectories",
    ]
    for label, ax, title in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d], panel_titles):
        ax.text(-0.16, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top", ha="left")
        ax.set_title(title, loc="left", pad=2)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    for spine in ax_d2.spines.values():
        spine.set_linewidth(0.6)
    ax_d2.tick_params(width=0.6, length=2.5, labelsize=6)

    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    outdir.mkdir(parents=True, exist_ok=True)
    fig.savefig(outdir / "fig_nk_local_channel_index.pdf")
    fig.savefig(outdir / "fig_nk_local_channel_index.png", dpi=450)
    plt.close(fig)


def write_channel_numbers(pred: pd.DataFrame, binned: pd.DataFrame, local_future: pd.DataFrame, outpath: Path) -> None:
    """Write LaTeX macros for the manuscript text."""
    def r2(model: str, h: int) -> float:
        row = pred[(pred["model"] == model) & (pred["horizon"] == h)]
        return float(row["r2_test"].iloc[0])

    def coef_a(model: str, h: int) -> float:
        row = pred[(pred["model"] == model) & (pred["horizon"] == h)]
        return float(row["coef_channel_index"].iloc[0])

    b = binned.sort_values("decile")
    low = float(b["mean_future_gain"].iloc[0])
    high = float(b["mean_future_gain"].iloc[-1])
    diff = high - low

    corr = float(
        local_future[["local_channel_index", "future_gain_h5"]]
        .dropna()
        .corr()
        .iloc[0, 1]
    )
    text = f"""% Automatically generated by nk_local_channel_index.py
\\newcommand{{\\NKChannelRtwoBaselineHFive}}{{{r2('baseline', 5):.3f}}}
\\newcommand{{\\NKChannelRtwoIndexHFive}}{{{r2('baseline_plus_index', 5):.3f}}}
\\newcommand{{\\NKChannelRtwoPlusHFive}}{{{r2('baseline_plus_channel', 5):.3f}}}
\\newcommand{{\\NKChannelRtwoGainHFive}}{{{(r2('baseline_plus_index', 5)-r2('baseline', 5)):.3f}}}
\\newcommand{{\\NKChannelRtwoFullGainHFive}}{{{(r2('baseline_plus_channel', 5)-r2('baseline', 5)):.3f}}}
\\newcommand{{\\NKChannelCoefIndexHFive}}{{{coef_a('baseline_plus_index', 5):.4f}}}
\\newcommand{{\\NKChannelLowBinGainHFive}}{{{low:.4f}}}
\\newcommand{{\\NKChannelHighBinGainHFive}}{{{high:.4f}}}
\\newcommand{{\\NKChannelHighLowDiffHFive}}{{{diff:.4f}}}
\\newcommand{{\\NKChannelCorrHFive}}{{{corr:.3f}}}
"""
    outpath.write_text(text)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    fig_dir = base / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    channel_params = ChannelSamplingParameters(n_channel_samples=96, horizons=(1, 3, 5, 10))
    trajectories, local = run_channel_grid(channel_params=channel_params)
    local_future = add_future_gains(trajectories, local, channel_params.horizons)
    pred = prediction_table(local_future, channel_params.horizons)
    binned = bin_local_channel_index(local_future, horizon=5, n_bins=10)

    trajectories.to_csv(data_dir / "nk_channel_trajectories.csv", index=False)
    local.to_csv(data_dir / "nk_local_channel_index.csv", index=False)
    local_future.to_csv(data_dir / "nk_local_channel_with_future_gains.csv", index=False)
    pred.to_csv(data_dir / "nk_local_channel_prediction.csv", index=False)
    binned.to_csv(data_dir / "nk_local_channel_index_binned_h5.csv", index=False)

    plot_four_panel_nk_summary(base)
    plot_local_channel_results(local_future, pred, binned, fig_dir)
    write_channel_numbers(pred, binned, local_future, base / "nk_local_channel_numbers.tex")

    print("Prediction summary:")
    print(pred.to_string(index=False))
    print("\nBinned h=5 summary:")
    print(binned.to_string(index=False))
    print("\nWrote figures and data to", base)


if __name__ == "__main__":
    main()
