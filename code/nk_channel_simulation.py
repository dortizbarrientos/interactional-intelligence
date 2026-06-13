"""
NK-channel simulation for the interactional-intelligence manuscript.

This is an illustrative model, not a psychological model of people or LLMs.
It tests a simple channel claim on a controlled landscape:

    Coupling helps when broad proposal generation, useful evaluation, and
    stabilisation are aligned; it can hurt when proposal breadth exceeds
    evaluation reliability on rugged landscapes.

State space
-----------
An artefact is a binary string of length N. Its quality is defined by a
Kauffman NK landscape. Increasing K increases epistatic ruggedness.

Agents
------
H          Human-like local search: narrow proposals, relatively accurate evaluation.
M          Model-like search: broader proposals, noisier and biased evaluation.
HM_naive   Human + model proposals, human-like evaluation, no uncertainty margin.
HM_channel Human + model proposals, human-like evaluation, uncertainty margin.

Strong complementarity
----------------------
For each paired landscape/start/D setting:

    Delta_strong = Q(HM_channel) - max(Q(H), Q(M))

A positive value means the stabilised coupled system beats both solo components.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import itertools
import math
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# Landscape
# -----------------------------


class NKLandscape:
    """NK landscape with integer-encoded binary states."""

    def __init__(self, N: int, K: int, rng: np.random.Generator):
        if K < 0 or K >= N:
            raise ValueError("K must satisfy 0 <= K < N")
        self.N = N
        self.K = K
        self.rng = rng
        self.partners: List[np.ndarray] = []

        for i in range(N):
            others = [j for j in range(N) if j != i]
            if K == 0:
                idx = [i]
            else:
                idx = [i] + list(rng.choice(others, size=K, replace=False))
            self.partners.append(np.asarray(idx, dtype=np.uint8))

        self.tables = [rng.random(1 << (K + 1)) for _ in range(N)]
        self.fitness_array = self._compute_fitness_array()

    def _compute_fitness_array(self) -> np.ndarray:
        """Precompute quality for all 2^N states."""
        num_states = 1 << self.N
        states = np.arange(num_states, dtype=np.uint32)
        fitness = np.zeros(num_states, dtype=np.float64)

        for idxs, table in zip(self.partners, self.tables):
            table_index = np.zeros(num_states, dtype=np.uint32)
            for local_bit, global_bit in enumerate(idxs):
                bit = ((states >> int(global_bit)) & 1).astype(np.uint32)
                table_index |= bit << local_bit
            fitness += table[table_index]

        return fitness / self.N

    def fitness(self, state: int) -> float:
        return float(self.fitness_array[int(state)])


# -----------------------------
# Proposal masks and samplers
# -----------------------------


def precompute_masks(N: int, max_distance: int) -> Dict[int, np.ndarray]:
    """All bit-flip masks of each Hamming distance up to max_distance."""
    masks: Dict[int, np.ndarray] = {}
    for d in range(1, max_distance + 1):
        vals = []
        for bits in itertools.combinations(range(N), d):
            mask = 0
            for b in bits:
                mask |= 1 << b
            vals.append(mask)
        masks[d] = np.asarray(vals, dtype=np.uint32)
    return masks


def draw_human_distance(rng: np.random.Generator) -> int:
    """Mostly local proposals, with occasional two- or three-bit changes."""
    r = rng.random()
    if r < 0.85:
        return 1
    if r < 0.97:
        return 2
    return 3


def draw_model_distance(rng: np.random.Generator, D: int, cdf_cache: Dict[int, np.ndarray]) -> int:
    """Broader proposals. D is the maximum Hamming distance sampled."""
    if D <= 1:
        return 1
    cdf = cdf_cache[D]
    return int(np.searchsorted(cdf, rng.random(), side="right") + 1)


def draw_mask(rng: np.random.Generator, masks: Dict[int, np.ndarray], d: int) -> int:
    arr = masks[d]
    return int(arr[int(rng.integers(0, len(arr)))])


# -----------------------------
# Evaluation and adaptive walk
# -----------------------------


def evaluation_sd(kind: str, K: int, N: int, distance: int) -> float:
    """Noise in estimating a proposal's quality change.

    The human-like evaluator is more accurate, but the uncertainty grows
    with ruggedness and proposal distance. This implements the idea that
    broad changes are harder to judge on curved landscapes.
    """
    ruggedness = K / max(1, N - 1)
    if kind == "human":
        return 0.010 * (1.0 + 2.8 * ruggedness * distance)
    if kind == "model":
        return 0.035 * (1.0 + 1.8 * ruggedness * distance)
    raise ValueError(f"Unknown evaluator kind: {kind}")


@dataclass(frozen=True)
class AgentParameters:
    n_human_proposals: int = 4
    n_model_proposals: int = 12
    model_bias_strength: float = 0.012
    channel_margin_lambda: float = 0.70


def simulate_condition(
    landscape: NKLandscape,
    start: int,
    condition: str,
    steps: int,
    D: int,
    masks: Dict[int, np.ndarray],
    cdf_cache: Dict[int, np.ndarray],
    rng: np.random.Generator,
    params: AgentParameters,
) -> Dict[str, object]:
    """Run one adaptive walk for one condition."""
    state = int(start)
    fitness = landscape.fitness(state)
    trajectory = [fitness]
    accepted_total = 0
    accepted_bad = 0
    accepted_distances: List[int] = []

    for _ in range(steps):
        proposals: List[Tuple[int, int, str]] = []  # new state, distance, source

        if condition == "H":
            for _ in range(params.n_human_proposals):
                d = draw_human_distance(rng)
                proposals.append((state ^ draw_mask(rng, masks, d), d, "human"))
            evaluator = "human"
            bias_strength = 0.0
            margin_lambda = 0.0

        elif condition == "M":
            for _ in range(params.n_model_proposals):
                d = draw_model_distance(rng, D, cdf_cache)
                proposals.append((state ^ draw_mask(rng, masks, d), d, "model"))
            evaluator = "model"
            bias_strength = params.model_bias_strength
            margin_lambda = 0.0

        elif condition == "HM_naive":
            for _ in range(params.n_human_proposals):
                d = draw_human_distance(rng)
                proposals.append((state ^ draw_mask(rng, masks, d), d, "human"))
            for _ in range(params.n_model_proposals):
                d = draw_model_distance(rng, D, cdf_cache)
                proposals.append((state ^ draw_mask(rng, masks, d), d, "model"))
            evaluator = "human"
            bias_strength = 0.0
            margin_lambda = 0.0

        elif condition == "HM_channel":
            for _ in range(params.n_human_proposals):
                d = draw_human_distance(rng)
                proposals.append((state ^ draw_mask(rng, masks, d), d, "human"))
            for _ in range(params.n_model_proposals):
                d = draw_model_distance(rng, D, cdf_cache)
                proposals.append((state ^ draw_mask(rng, masks, d), d, "model"))
            evaluator = "human"
            bias_strength = 0.0
            margin_lambda = params.channel_margin_lambda

        else:
            raise ValueError(f"Unknown condition: {condition}")

        best_estimate = -math.inf
        best = None

        for new_state, distance, source in proposals:
            new_fitness = landscape.fitness(new_state)
            true_delta = new_fitness - fitness
            sd = evaluation_sd(evaluator, landscape.K, landscape.N, distance)

            # The model-alone condition overvalues broad proposals. This is a
            # toy representation of false fluency, not a claim about any system.
            bias = bias_strength * (distance - 1) * (1.0 + landscape.K / max(1, landscape.N - 1))
            perceived_delta = true_delta + rng.normal(0.0, sd) + bias

            if perceived_delta > best_estimate:
                best_estimate = perceived_delta
                best = (new_state, new_fitness, distance, true_delta, sd, source, perceived_delta)

        assert best is not None
        new_state, new_fitness, distance, true_delta, sd, source, perceived_delta = best

        # The stabilised channel accepts only changes that clear an uncertainty
        # margin. Other conditions accept any perceived improvement.
        if perceived_delta > margin_lambda * sd:
            state = new_state
            fitness = new_fitness
            accepted_total += 1
            accepted_bad += int(true_delta < 0.0)
            accepted_distances.append(distance)

        trajectory.append(fitness)

    return {
        "final_fitness": fitness,
        "trajectory": np.asarray(trajectory, dtype=float),
        "accepted_total": accepted_total,
        "accepted_bad": accepted_bad,
        "accepted_bad_fraction": accepted_bad / accepted_total if accepted_total else 0.0,
        "accepted_mean_distance": float(np.mean(accepted_distances)) if accepted_distances else 0.0,
    }


# -----------------------------
# Simulation grid and plotting
# -----------------------------


def run_grid(
    N: int = 16,
    K_values: Iterable[int] = (0, 2, 4, 8, 12),
    D_values: Iterable[int] = (1, 2, 4, 8),
    reps: int = 80,
    steps: int = 50,
    seed: int = 20260611,
    params: AgentParameters = AgentParameters(),
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Run paired simulations across landscapes, K and proposal breadth."""
    K_values = list(K_values)
    D_values = list(D_values)
    max_D = max(D_values)
    masks = precompute_masks(N, max_D)

    cdf_cache: Dict[int, np.ndarray] = {}
    for D in D_values:
        weights = np.arange(1, D + 1, dtype=float)
        weights /= weights.sum()
        cdf_cache[D] = np.cumsum(weights)

    condition_seed_offsets = {"H": 101, "M": 202, "HM_naive": 303, "HM_channel": 404}
    rng_main = np.random.default_rng(seed)
    rows = []
    traj_rows = []

    for K in K_values:
        for rep in range(reps):
            landscape_seed = int(rng_main.integers(1, 2_000_000_000))
            landscape_rng = np.random.default_rng(landscape_seed)
            landscape = NKLandscape(N, K, landscape_rng)
            start = int(landscape_rng.integers(0, 1 << N, dtype=np.uint32))
            start_fitness = landscape.fitness(start)

            for D in D_values:
                finals: Dict[str, float] = {}

                for condition in ("H", "M", "HM_naive", "HM_channel"):
                    agent_seed = landscape_seed + condition_seed_offsets[condition] + 1009 * D
                    agent_rng = np.random.default_rng(agent_seed)
                    out = simulate_condition(
                        landscape=landscape,
                        start=start,
                        condition=condition,
                        steps=steps,
                        D=D,
                        masks=masks,
                        cdf_cache=cdf_cache,
                        rng=agent_rng,
                        params=params,
                    )
                    finals[condition] = float(out["final_fitness"])
                    rows.append(
                        {
                            "N": N,
                            "K": K,
                            "D": D,
                            "rep": rep,
                            "condition": condition,
                            "start_fitness": start_fitness,
                            "final_fitness": out["final_fitness"],
                            "improvement": out["final_fitness"] - start_fitness,
                            "accepted_total": out["accepted_total"],
                            "accepted_bad": out["accepted_bad"],
                            "accepted_bad_fraction": out["accepted_bad_fraction"],
                            "accepted_mean_distance": out["accepted_mean_distance"],
                            "landscape_seed": landscape_seed,
                            "steps": steps,
                        }
                    )

                    # Keep only a subset of trajectories for compact plotting.
                    if rep < 12 and D in (1, 4, 8) and K in (0, 4, 12):
                        for step, value in enumerate(out["trajectory"]):
                            traj_rows.append(
                                {
                                    "N": N,
                                    "K": K,
                                    "D": D,
                                    "rep": rep,
                                    "condition": condition,
                                    "step": step,
                                    "fitness": value,
                                    "landscape_seed": landscape_seed,
                                }
                            )

                rows.append(
                    {
                        "N": N,
                        "K": K,
                        "D": D,
                        "rep": rep,
                        "condition": "Strong_delta_HM_channel",
                        "start_fitness": start_fitness,
                        "final_fitness": finals["HM_channel"] - max(finals["H"], finals["M"]),
                        "improvement": np.nan,
                        "accepted_total": np.nan,
                        "accepted_bad": np.nan,
                        "accepted_bad_fraction": np.nan,
                        "accepted_mean_distance": np.nan,
                        "landscape_seed": landscape_seed,
                        "steps": steps,
                    }
                )
                rows.append(
                    {
                        "N": N,
                        "K": K,
                        "D": D,
                        "rep": rep,
                        "condition": "Strong_delta_HM_naive",
                        "start_fitness": start_fitness,
                        "final_fitness": finals["HM_naive"] - max(finals["H"], finals["M"]),
                        "improvement": np.nan,
                        "accepted_total": np.nan,
                        "accepted_bad": np.nan,
                        "accepted_bad_fraction": np.nan,
                        "accepted_mean_distance": np.nan,
                        "landscape_seed": landscape_seed,
                        "steps": steps,
                    }
                )

    return pd.DataFrame(rows), pd.DataFrame(traj_rows)


def sem(series: pd.Series) -> float:
    return float(series.std(ddof=1) / math.sqrt(len(series)))


def summarise(results: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    condition_results = results[~results["condition"].str.startswith("Strong")].copy()
    summary = (
        condition_results.groupby(["K", "D", "condition"], as_index=False)
        .agg(
            mean_final=("final_fitness", "mean"),
            sem_final=("final_fitness", sem),
            mean_improvement=("improvement", "mean"),
            mean_bad_acceptance=("accepted_bad_fraction", "mean"),
            mean_accepted_distance=("accepted_mean_distance", "mean"),
            n=("final_fitness", "size"),
        )
    )
    deltas = (
        results[results["condition"].str.startswith("Strong")]
        .groupby(["K", "D", "condition"], as_index=False)
        .agg(
            mean_delta=("final_fitness", "mean"),
            sem_delta=("final_fitness", sem),
            p_positive=("final_fitness", lambda x: float((x > 0).mean())),
            n=("final_fitness", "size"),
        )
    )
    params = results[["N", "steps"]].drop_duplicates().copy()
    return summary, deltas, params


def plot_final_by_condition(summary: pd.DataFrame, outdir: Path, D: int = 4) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    subset = summary[summary["D"] == D]
    for condition in ["H", "M", "HM_naive", "HM_channel"]:
        s = subset[subset["condition"] == condition].sort_values("K")
        ax.errorbar(s["K"], s["mean_final"], yerr=s["sem_final"], marker="o", linewidth=1.6, capsize=3, label=condition)
    ax.set_xlabel("Landscape ruggedness, K")
    ax.set_ylabel("Final quality")
    ax.set_title(f"Coupled search on NK landscapes (D = {D})")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "fig_nk_final_by_condition.pdf")
    fig.savefig(outdir / "fig_nk_final_by_condition.png", dpi=300)
    plt.close(fig)


def plot_phase_diagram(deltas: pd.DataFrame, outdir: Path, condition: str = "Strong_delta_HM_channel") -> None:
    subset = deltas[deltas["condition"] == condition]
    table = subset.pivot(index="K", columns="D", values="mean_delta").sort_index()
    fig, ax = plt.subplots(figsize=(5.6, 4.1))
    im = ax.imshow(table.values, aspect="auto", origin="lower")
    ax.set_xticks(np.arange(table.shape[1]), labels=[str(c) for c in table.columns])
    ax.set_yticks(np.arange(table.shape[0]), labels=[str(i) for i in table.index])
    ax.set_xlabel("Model proposal breadth, D")
    ax.set_ylabel("Landscape ruggedness, K")
    ax.set_title("Strong complementarity of stabilised coupling")
    for i, K in enumerate(table.index):
        for j, D in enumerate(table.columns):
            val = table.loc[K, D]
            ax.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=8)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label("Mean Delta strong")
    fig.tight_layout()
    fig.savefig(outdir / "fig_nk_phase_diagram.pdf")
    fig.savefig(outdir / "fig_nk_phase_diagram.png", dpi=300)
    plt.close(fig)


def plot_trajectory(trajectories: pd.DataFrame, outdir: Path, K: int = 4, D: int = 4) -> None:
    subset = trajectories[(trajectories["K"] == K) & (trajectories["D"] == D)]
    mean_traj = (
        subset.groupby(["condition", "step"], as_index=False)
        .agg(mean_fitness=("fitness", "mean"), sem_fitness=("fitness", sem))
    )
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    for condition in ["H", "M", "HM_naive", "HM_channel"]:
        s = mean_traj[mean_traj["condition"] == condition].sort_values("step")
        ax.plot(s["step"], s["mean_fitness"], linewidth=1.7, label=condition)
    ax.set_xlabel("Step")
    ax.set_ylabel("Quality")
    ax.set_title(f"Mean trajectories in an intermediate landscape (K = {K}, D = {D})")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "fig_nk_trajectory_K4_D4.pdf")
    fig.savefig(outdir / "fig_nk_trajectory_K4_D4.png", dpi=300)
    plt.close(fig)


def plot_bad_acceptance(summary: pd.DataFrame, outdir: Path, D: int = 8) -> None:
    subset = summary[(summary["D"] == D) & (summary["condition"].isin(["HM_naive", "HM_channel"]))]
    fig, ax = plt.subplots(figsize=(5.8, 4.1))
    for condition in ["HM_naive", "HM_channel"]:
        s = subset[subset["condition"] == condition].sort_values("K")
        ax.plot(s["K"], s["mean_bad_acceptance"], marker="o", linewidth=1.7, label=condition)
    ax.set_xlabel("Landscape ruggedness, K")
    ax.set_ylabel("Fraction of accepted moves that lower true quality")
    ax.set_title(f"Accepted harmful moves under broad proposals (D = {D})")
    ax.legend(frameon=False)
    ax.grid(True, linewidth=0.3, alpha=0.4)
    fig.tight_layout()
    fig.savefig(outdir / "fig_nk_bad_acceptance.pdf")
    fig.savefig(outdir / "fig_nk_bad_acceptance.png", dpi=300)
    plt.close(fig)


def write_manuscript_numbers(deltas: pd.DataFrame, summary: pd.DataFrame, outpath: Path) -> None:
    channel = deltas[deltas["condition"] == "Strong_delta_HM_channel"]
    def get_delta(K: int, D: int) -> float:
        row = channel[(channel["K"] == K) & (channel["D"] == D)]
        return float(row["mean_delta"].iloc[0])
    def get_p(K: int, D: int) -> float:
        row = channel[(channel["K"] == K) & (channel["D"] == D)]
        return float(row["p_positive"].iloc[0])
    def get_bad(condition: str, K: int, D: int) -> float:
        row = summary[(summary["condition"] == condition) & (summary["K"] == K) & (summary["D"] == D)]
        return float(row["mean_bad_acceptance"].iloc[0])

    text = f"""% Automatically generated by nk_channel_simulation.py
\\newcommand{{\\NKDeltaSmooth}}{{{get_delta(0,4):.3f}}}
\\newcommand{{\\NKDeltaIntermediate}}{{{get_delta(4,4):.3f}}}
\\newcommand{{\\NKDeltaHighBroad}}{{{get_delta(12,8):.3f}}}
\\newcommand{{\\NKPositiveIntermediate}}{{{get_p(4,4):.2f}}}
\\newcommand{{\\NKBadNaiveHighBroad}}{{{get_bad('HM_naive',12,8):.2f}}}
\\newcommand{{\\NKBadChannelHighBroad}}{{{get_bad('HM_channel',12,8):.2f}}}
"""
    outpath.write_text(text)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    data_dir = base / "data"
    fig_dir = base / "figures"
    data_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    results, trajectories = run_grid()
    summary, deltas, params = summarise(results)

    results.to_csv(data_dir / "nk_results.csv", index=False)
    trajectories.to_csv(data_dir / "nk_trajectories_subset.csv", index=False)
    summary.to_csv(data_dir / "nk_summary_by_condition.csv", index=False)
    deltas.to_csv(data_dir / "nk_strong_complementarity.csv", index=False)
    params.to_csv(data_dir / "nk_parameter_summary.csv", index=False)

    plot_final_by_condition(summary, fig_dir, D=4)
    plot_phase_diagram(deltas, fig_dir)
    plot_trajectory(trajectories, fig_dir, K=4, D=4)
    plot_bad_acceptance(summary, fig_dir, D=8)
    write_manuscript_numbers(deltas, summary, base / "nk_manuscript_numbers.tex")

    print("Wrote results to", data_dir)
    print("Wrote figures to", fig_dir)
    print(deltas[deltas["condition"] == "Strong_delta_HM_channel"].pivot(index="K", columns="D", values="mean_delta"))


if __name__ == "__main__":
    main()
