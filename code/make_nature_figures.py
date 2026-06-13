"""Create Nature-style vector figures for the interactional-intelligence manuscript."""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Ellipse, Circle, PathPatch
from matplotlib.colors import TwoSlopeNorm


def style() -> None:
    mpl.rcParams.update({
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
    })

COL = {
    "H": "#0072B2",          # blue
    "M": "#E69F00",          # orange
    "HM_naive": "#009E73",   # bluish green
    "HM_channel": "#000000", # black
    "grey": "#666666",
    "light": "#f4f4f4",
    "line": "#333333",
    "accent": "#CC79A7",     # reddish purple
}


def panel_label(ax, label: str) -> None:
    ax.text(-0.08, 1.06, label, transform=ax.transAxes, fontweight="bold", fontsize=8, va="top", ha="left")


def box(ax, xy, w, h, text, fc="white", ec="#333333", lw=0.8, fontsize=7, rounded=True):
    style_str = "round,pad=0.025,rounding_size=0.025" if rounded else "square,pad=0.02"
    patch = FancyBboxPatch(xy, w, h, boxstyle=style_str, fc=fc, ec=ec, lw=lw)
    ax.add_patch(patch)
    ax.text(xy[0] + w/2, xy[1] + h/2, text, ha="center", va="center", fontsize=fontsize)
    return patch


def arrow(ax, xy1, xy2, color="#333333", lw=0.8, rad=0.0, style="-|>"):
    patch = FancyArrowPatch(xy1, xy2, arrowstyle=style, mutation_scale=8, lw=lw, color=color,
                            connectionstyle=f"arc3,rad={rad}")
    ax.add_patch(patch)
    return patch


def make_concept_figure(base: Path) -> None:
    style()
    fig, axes = plt.subplots(2, 2, figsize=(7.20, 4.90))
    axes = axes.ravel()

    # a. Coupled system loop.
    ax = axes[0]
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel_label(ax, "a")
    ax.set_title("The coupled cognitive unit", loc="left", pad=2)
    box(ax, (0.07, 0.56), 0.24, 0.16, "Human\ngoals, judgement", fc="#eef6fb", ec=COL["H"])
    box(ax, (0.69, 0.56), 0.24, 0.16, "Model\nproposal breadth", fc="#fff4df", ec=COL["M"])
    box(ax, (0.37, 0.40), 0.26, 0.16, "Shared artefact\n$x_t$", fc="#f3f3f3", ec="#222222")
    box(ax, (0.37, 0.12), 0.26, 0.16, "Task environment\n$Q(x)$", fc="#f8f8f8", ec=COL["grey"])
    box(ax, (0.37, 0.72), 0.26, 0.12, "Interface + memory", fc="#f8f1f6", ec=COL["accent"], fontsize=6.5)
    arrow(ax, (0.31, 0.64), (0.37, 0.51), COL["H"])
    arrow(ax, (0.69, 0.64), (0.63, 0.51), COL["M"])
    arrow(ax, (0.50, 0.40), (0.50, 0.28), COL["line"])
    arrow(ax, (0.50, 0.28), (0.50, 0.40), COL["line"], rad=0.35)
    arrow(ax, (0.50, 0.72), (0.50, 0.56), COL["accent"])
    arrow(ax, (0.37, 0.78), (0.31, 0.70), COL["accent"], rad=0.15)
    arrow(ax, (0.63, 0.78), (0.69, 0.70), COL["accent"], rad=-0.15)
    ax.text(0.50, 0.33, "feedback", ha="center", va="center", fontsize=6, color=COL["grey"])

    # b. Channel geometry.
    ax = axes[1]
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel_label(ax, "b")
    ax.set_title("Channel geometry", loc="left", pad=2)
    # Landscape contours.
    for w, h, alpha in [(0.72, 0.44, 0.16), (0.52, 0.30, 0.22), (0.32, 0.17, 0.30)]:
        ax.add_patch(Ellipse((0.52, 0.52), w, h, angle=25, fill=False, lw=0.8, ec=str(0.7-alpha)))
    ax.add_patch(Ellipse((0.45, 0.47), 0.42, 0.16, angle=25, fill=True, fc="#e7f2f8", ec=COL["H"], lw=0.9, alpha=0.9))
    arrow(ax, (0.32, 0.38), (0.70, 0.72), "#000000", lw=1.0)
    ax.text(0.71, 0.73, "$g_t$", fontsize=8, va="center")
    arrow(ax, (0.45, 0.47), (0.68, 0.59), COL["H"], lw=1.0)
    ax.text(0.39, 0.31, "proposal\noperator $J_t$", ha="center", va="top", fontsize=6.5, color=COL["H"])
    arrow(ax, (0.45, 0.47), (0.49, 0.70), COL["accent"], lw=0.9)
    ax.text(0.50, 0.71, "curvature\n$H_t$", fontsize=6.5, color=COL["accent"], va="bottom")
    ax.text(0.05, 0.09, r"$A^I_t=\dfrac{g_t^\top J_t g_t}{g_t^\top J_t g_t+\lambda\,\mathrm{tr}(S_t^2)+\rho U_t}$", fontsize=7)
    ax.text(0.56, 0.31, r"$S_t=J_t^{1/2}H_tJ_t^{1/2}$", fontsize=7)

    # c. Branching conversations.
    ax = axes[2]
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel_label(ax, "c")
    ax.set_title("Branch the trajectory", loc="left", pad=2)
    box(ax, (0.36, 0.72), 0.28, 0.13, "Freeze artefact\n$x_t$", fc="#f3f3f3", ec="#222222", fontsize=6.5)
    endpoints = [
        (0.08, 0.43, "human\nalone", COL["H"]),
        (0.27, 0.22, "model\nalone", COL["M"]),
        (0.47, 0.43, "same\npair", "#000000"),
        (0.66, 0.22, "new\nexpert", COL["accent"]),
        (0.82, 0.43, "relay or\nagents", COL["grey"]),
    ]
    for x, y, txt, c in endpoints:
        box(ax, (x, y), 0.14, 0.12, txt, fc="white", ec=c, fontsize=6.2)
        arrow(ax, (0.50, 0.72), (x+0.07, y+0.12), c, lw=0.8, rad=0.05)
    box(ax, (0.36, 0.02), 0.28, 0.12, "blind scoring\n$Q(x_{t+h})$", fc="#f8f8f8", ec=COL["grey"], fontsize=6.5)
    for x, y, txt, c in endpoints:
        arrow(ax, (x+0.07, y), (0.50, 0.14), c, lw=0.65, rad=0.03)

    # d. Observable map.
    ax = axes[3]
    ax.set_axis_off(); ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    panel_label(ax, "d")
    ax.set_title("From theory to observables", loc="left", pad=2)
    rows = [
        ("$x_t$", "artefact state"),
        ("$J_t$", "sampled alternative moves"),
        ("$g_t$", "local preference direction"),
        ("$H_t$", "failure of additivity"),
        ("$A_t$", "predicted near-term gain"),
    ]
    y0 = 0.80
    for i, (sym, desc) in enumerate(rows):
        y = y0 - i*0.15
        box(ax, (0.08, y-0.055), 0.16, 0.10, sym, fc="#f3f3f3", ec="#333333", fontsize=7)
        arrow(ax, (0.24, y-0.005), (0.34, y-0.005), COL["grey"], lw=0.7)
        box(ax, (0.35, y-0.055), 0.52, 0.10, desc, fc="white", ec="#999999", fontsize=6.5)
    ax.text(0.08, 0.05, "The claim is testable when each symbol has a measurement.", fontsize=6.7, color=COL["grey"])

    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    out = base / "figures" / "fig1_conceptual_framework.pdf"
    fig.savefig(out)
    fig.savefig(base / "figures" / "fig1_conceptual_framework.png", dpi=450)
    plt.close(fig)


def make_main_nk_figure(base: Path) -> None:
    style()
    data_dir = base / "data"
    deltas = pd.read_csv(data_dir / "nk_strong_complementarity.csv")
    binned = pd.read_csv(data_dir / "nk_local_channel_index_binned_h5.csv")
    pred = pd.read_csv(data_dir / "nk_local_channel_prediction.csv")
    local_future = pd.read_csv(data_dir / "nk_local_channel_with_future_gains.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.05))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    # a. Phase diagram.
    delta_subset = deltas[deltas["condition"] == "Strong_delta_HM_channel"]
    table = delta_subset.pivot(index="K", columns="D", values="mean_delta").sort_index()
    vmax = max(abs(float(table.values.min())), abs(float(table.values.max())))
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax)
    im = ax_a.imshow(table.values, origin="lower", aspect="auto", cmap="PuOr_r", norm=norm)
    ax_a.set_xticks(np.arange(table.shape[1]), labels=[str(c) for c in table.columns])
    ax_a.set_yticks(np.arange(table.shape[0]), labels=[str(i) for i in table.index])
    ax_a.set_xlabel("Model proposal breadth, D")
    ax_a.set_ylabel("Landscape ruggedness, K")
    for i, K in enumerate(table.index):
        for j, D in enumerate(table.columns):
            val = table.loc[K, D]
            ax_a.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5.3, color="white" if abs(val) > 0.055 else "black")
    cbar = fig.colorbar(im, ax=ax_a, fraction=0.046, pad=0.02)
    cbar.set_label(r"Mean $\Delta_{strong}$")
    cbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)

    # b. Decile relationship.
    ax_b.errorbar(
        binned["decile"], binned["mean_future_gain"], yerr=binned["sem_future_gain"],
        marker="o", color="black", linewidth=0.9)
    ax_b.axhline(0, color="#777777", linewidth=0.6)
    ax_b.set_xlabel(r"Local channel index decile, $A^{NK}_t$")
    ax_b.set_ylabel(r"Realised gain, $Q_{t+5}-Q_t$")
    ax_b.set_xticks([1, 5, 10])

    # c. R2 across horizons.
    wide = pred.pivot(index="horizon", columns="model", values="r2_test").reset_index()
    ax_c.plot(wide["horizon"], wide["baseline"], marker="o", color="#777777", label="baseline")
    ax_c.plot(wide["horizon"], wide["baseline_plus_index"], marker="o", color=COL["H"], label=r"+ $A^{NK}_t$")
    ax_c.plot(wide["horizon"], wide["baseline_plus_channel"], marker="o", color="black", label="+ channel terms")
    ax_c.set_xlabel("Prediction horizon, h")
    ax_c.set_ylabel(r"Held-out $R^2$")
    ax_c.set_xticks(list(wide["horizon"]))
    ax_c.legend(frameon=False, loc="lower right", handlelength=1.4)

    # d. Trajectory quality and channel index.
    subset = local_future[(local_future["K"] == 4) & (local_future["D"] == 4)]
    mean_by_step = subset.groupby("step", as_index=False).agg(
        mean_quality=("quality", "mean"),
        mean_channel=("local_channel_index", "mean"),
        sem_channel=("local_channel_index", lambda x: x.std(ddof=1)/np.sqrt(len(x)))
    )
    ax_d.plot(mean_by_step["step"], mean_by_step["mean_quality"], color="black", label="quality")
    ax_d.set_xlabel("Step")
    ax_d.set_ylabel("Quality")
    ax_d2 = ax_d.twinx()
    ax_d2.plot(mean_by_step["step"], mean_by_step["mean_channel"], color=COL["H"], label=r"$A^{NK}_t$")
    ax_d2.set_ylabel(r"Local channel index")
    lines, labels = ax_d.get_legend_handles_labels()
    lines2, labels2 = ax_d2.get_legend_handles_labels()
    ax_d.legend(lines + lines2, labels + labels2, frameon=False, loc="center right", handlelength=1.4)
    ax_d2.tick_params(width=0.6, length=2.5, labelsize=6)
    for spine in ax_d2.spines.values():
        spine.set_linewidth(0.6)

    titles = [
        "Strong complementarity is conditional",
        "Local channel index predicts gain",
        "Channel terms improve prediction",
        "Channels close along a trajectory",
    ]
    for label, ax, title in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d], titles):
        panel_label(ax, label)
        ax.set_title(title, loc="left", pad=2)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    fig.savefig(base / "figures" / "fig2_nk_local_channel.pdf")
    fig.savefig(base / "figures" / "fig2_nk_local_channel.png", dpi=450)
    plt.close(fig)


def make_extended_nk_summary(base: Path) -> None:
    style()
    data_dir = base / "data"
    outdir = base / "extended_data"
    summary = pd.read_csv(data_dir / "nk_summary_by_condition.csv")
    deltas = pd.read_csv(data_dir / "nk_strong_complementarity.csv")
    trajectories = pd.read_csv(data_dir / "nk_trajectories_subset.csv")

    fig, axes = plt.subplots(2, 2, figsize=(7.20, 5.05))
    ax_a, ax_b, ax_c, ax_d = axes.ravel()

    subset = summary[summary["D"] == 4]
    for condition in ["H", "M", "HM_naive", "HM_channel"]:
        s = subset[subset["condition"] == condition].sort_values("K")
        ax_a.errorbar(s["K"], s["mean_final"], yerr=s["sem_final"], marker="o", linewidth=0.9,
                      label=condition, color=COL[condition])
    ax_a.set_xlabel("Landscape ruggedness, K")
    ax_a.set_ylabel("Final quality")
    ax_a.set_xticks([0, 2, 4, 8, 12])
    ax_a.legend(frameon=False, loc="lower left", handlelength=1.4)

    subset_traj = trajectories[(trajectories["K"] == 4) & (trajectories["D"] == 4)]
    mean_traj = subset_traj.groupby(["condition", "step"], as_index=False).agg(mean_fitness=("fitness", "mean"))
    for condition in ["H", "M", "HM_naive", "HM_channel"]:
        s = mean_traj[mean_traj["condition"] == condition].sort_values("step")
        ax_b.plot(s["step"], s["mean_fitness"], label=condition, color=COL[condition])
    ax_b.set_xlabel("Step")
    ax_b.set_ylabel("Quality")
    ax_b.legend(frameon=False, loc="lower right", handlelength=1.4)

    delta_subset = deltas[deltas["condition"] == "Strong_delta_HM_channel"]
    table = delta_subset.pivot(index="K", columns="D", values="mean_delta").sort_index()
    vmax = max(abs(float(table.values.min())), abs(float(table.values.max())))
    im = ax_c.imshow(table.values, origin="lower", aspect="auto", cmap="PuOr_r", norm=TwoSlopeNorm(vcenter=0.0, vmin=-vmax, vmax=vmax))
    ax_c.set_xticks(np.arange(table.shape[1]), labels=[str(c) for c in table.columns])
    ax_c.set_yticks(np.arange(table.shape[0]), labels=[str(i) for i in table.index])
    ax_c.set_xlabel("Model proposal breadth, D")
    ax_c.set_ylabel("Landscape ruggedness, K")
    for i, K in enumerate(table.index):
        for j, D in enumerate(table.columns):
            val = table.loc[K, D]
            ax_c.text(j, i, f"{val:.3f}", ha="center", va="center", fontsize=5.3, color="white" if abs(val) > 0.055 else "black")
    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.02)
    cbar.set_label(r"Mean $\Delta_{strong}$")
    cbar.ax.tick_params(labelsize=5.5, width=0.5, length=2)

    subset_bad = summary[(summary["D"] == 8) & (summary["condition"].isin(["HM_naive", "HM_channel"]))]
    bad_colours = {"HM_naive": COL["H"], "HM_channel": COL["M"]}
    for condition in ["HM_naive", "HM_channel"]:
        s = subset_bad[subset_bad["condition"] == condition].sort_values("K")
        ax_d.plot(s["K"], s["mean_bad_acceptance"], marker="o", label=condition, color=bad_colours[condition])
    ax_d.set_xlabel("Landscape ruggedness, K")
    ax_d.set_ylabel("Accepted harmful moves")
    ax_d.set_xticks([0, 2, 4, 8, 12])
    ax_d.legend(frameon=False, loc="upper left", handlelength=1.4)

    titles = ["Final quality (D = 4)", "Trajectories (K = 4, D = 4)", "Strong complementarity", "Harmful accepted moves (D = 8)"]
    for label, ax, title in zip(["a", "b", "c", "d"], [ax_a, ax_b, ax_c, ax_d], titles):
        panel_label(ax, label)
        ax.set_title(title, loc="left", pad=2)
        ax.grid(False)
        for spine in ax.spines.values():
            spine.set_linewidth(0.6)
    fig.tight_layout(w_pad=1.0, h_pad=1.0)
    fig.savefig(outdir / "extended_data_fig1_nk_summary.pdf")
    fig.savefig(outdir / "extended_data_fig1_nk_summary.png", dpi=450)
    plt.close(fig)


def main() -> None:
    base = Path(__file__).resolve().parents[1]
    (base / "figures").mkdir(exist_ok=True)
    (base / "extended_data").mkdir(exist_ok=True)
    make_concept_figure(base)
    make_main_nk_figure(base)
    make_extended_nk_summary(base)
    print("Wrote manuscript figures.")


if __name__ == "__main__":
    main()
