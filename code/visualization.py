from __future__ import annotations

import argparse
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from rl_reward_project.envs import RewardMisspecGridworld


# ===============================
# COLORS (consistent palette)
# ===============================
COLORS = ["blue", "red", "lightgreen", "orange", "violet"]


# ===============================
# UTILITIES
# ===============================
def _moving_average(values: Sequence[float], window: int) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if window <= 1:
        return arr
    window = min(window, len(arr))
    kernel = np.ones(window) / window
    return np.convolve(arr, kernel, mode="same")


def _apply_grid(ax: plt.Axes) -> None:
    ax.grid(True, linewidth=0.5, alpha=0.3)


def _draw_grid_base(ax: plt.Axes, env: RewardMisspecGridworld) -> None:
    labels = env.grid_labels()
    rows, cols = labels.shape

    ax.set_xlim(-0.5, cols - 0.5)
    ax.set_ylim(rows - 0.5, -0.5)

    ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
    ax.grid(which="minor", linewidth=1)

    ax.set_xticks(range(cols))
    ax.set_yticks(range(rows))

    for r in range(rows):
        for c in range(cols):
            cell = env.cell_type((r, c))

            if cell == "wall":
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, alpha=0.6))

            elif cell == "cliff":
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, hatch="xx", fill=False)
                )

            elif cell == "risky":
                ax.add_patch(
                    plt.Rectangle((c - 0.5, r - 0.5), 1, 1, hatch="//", fill=False)
                )

            ax.text(c, r, labels[r, c], ha="center", va="center", fontsize=10)


# ===============================
# LEARNING CURVES
# ===============================
def plot_learning_curves(
    df: pd.DataFrame, output_path: str | Path, moving_average: int = 50
) -> None:
    episodes = df["episode"]

    # -------- RETURN --------
    fig, ax = plt.subplots(figsize=(9, 5))

    raw = df["episode_return"]
    smooth = _moving_average(raw, moving_average)

    ax.plot(episodes, raw, c="blue", alpha=0.4, label="Return")
    ax.plot(episodes, smooth, c="red", linewidth=2, label="Smoothed")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.set_title("Learning curve")
    ax.legend()
    _apply_grid(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    # -------- SUCCESS (SCATTER) --------
    fig, ax = plt.subplots(figsize=(9, 5))

    success = df["success"]

    ax.scatter(episodes, success, c="blue", s=10, alpha=0.6, label="Success (0/1)")

    # optional smoothed line (VERY useful visually)
    smooth_success = _moving_average(success, moving_average)
    ax.plot(episodes, smooth_success, c="red", linewidth=2, label="Smoothed")

    ax.set_xlabel("Episode")
    ax.set_ylabel("Success")
    ax.set_title("Success rate")
    ax.legend()
    _apply_grid(ax)

    fig.tight_layout()
    base = Path(output_path)
    fig.savefig(base.with_name("success_rate.png"), dpi=200)
    plt.close(fig)


# ===============================
# POLICY
# ===============================
def plot_grid_policy(
    env: RewardMisspecGridworld, q_table: np.ndarray, output_path: str | Path
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    _draw_grid_base(ax, env)

    policy = np.argmax(q_table, axis=1)

    for s in range(env.n_states):
        r, c = env.idx_to_state(s)

        if env.cell_type((r, c)) in {"wall", "goal", "cliff"}:
            continue

        ax.text(
            c,
            r,
            env.ACTION_SYMBOLS[int(policy[s])],
            ha="center",
            va="center",
            fontsize=16,
            color="blue",
        )

    ax.set_title("Greedy policy")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ===============================
# VALUE HEATMAP
# ===============================
def plot_value_heatmap(
    env: RewardMisspecGridworld, q_table: np.ndarray, output_path: str | Path
) -> None:
    values = np.max(q_table, axis=1).reshape(env.rows, env.cols)

    masked = values.astype(float).copy()
    for pos in env.walls:
        masked[pos] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(masked, cmap="plasma")

    _draw_grid_base(ax, env)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("State-value heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ===============================
# STATE VISITS
# ===============================
def plot_state_visits(
    env: RewardMisspecGridworld, state_visits: np.ndarray, output_path: str | Path
) -> None:
    visits = state_visits.reshape(env.rows, env.cols).astype(float)

    for pos in env.walls:
        visits[pos] = np.nan

    fig, ax = plt.subplots(figsize=(10, 6))

    im = ax.imshow(visits, cmap="plasma")

    _draw_grid_base(ax, env)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_title("State visit heatmap")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ===============================
# TRAJECTORY
# ===============================
def plot_trajectory(
    env: RewardMisspecGridworld,
    trajectory: Sequence[tuple[int, int]],
    output_path: str | Path,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6))

    _draw_grid_base(ax, env)

    xs = [c for r, c in trajectory]
    ys = [r for r, c in trajectory]

    ax.plot(xs, ys, marker="o", color="orange", linewidth=2)

    ax.set_title("Greedy trajectory")
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ===============================
# COMPARISON
# ===============================
def compare_metrics(
    metric_files: Sequence[str],
    labels: Sequence[str],
    metric: str,
    output_path: str | Path,
    moving_average: int = 50,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5))

    for i, (path, label) in enumerate(zip(metric_files, labels)):
        df = pd.read_csv(path)

        episodes = df["episode"]
        values = _moving_average(df[metric], moving_average)

        color = COLORS[i % len(COLORS)]

        ax.plot(episodes, values, label=label, color=color, linewidth=2)

    ax.set_xlabel("Episode")
    ax.set_ylabel(metric)
    ax.set_title(f"Comparison of {metric}")
    ax.legend()
    _apply_grid(ax)

    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


# ===============================
# CLI
# ===============================
def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Visualization utilities for the RL reward project."
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    compare = subparsers.add_parser("compare")
    compare.add_argument("--metrics", nargs="+", required=True)
    compare.add_argument("--labels", nargs="+", required=True)
    compare.add_argument("--metric", default="episode_return")
    compare.add_argument("--moving-average", type=int, default=50)
    compare.add_argument("--output", required=True)

    return parser


def main() -> None:
    parser = make_parser()
    args = parser.parse_args()

    if args.command == "compare":
        compare_metrics(
            args.metrics,
            args.labels,
            args.metric,
            args.output,
            args.moving_average,
        )


if __name__ == "__main__":
    main()
