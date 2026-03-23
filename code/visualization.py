from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

from envs.gridworld import RewardMisspecGridworld


ARROW_MAP = {0: "↑", 1: "→", 2: "↓", 3: "←"}


def moving_average(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or len(x) < window:
        return x.copy()
    kernel = np.ones(window) / window
    valid = np.convolve(x, kernel, mode="valid")
    prefix = np.full(window - 1, valid[0])
    return np.concatenate([prefix, valid])


def load_metrics(csv_path: Path) -> Dict[str, np.ndarray]:
    data = np.genfromtxt(csv_path, delimiter=",", names=True)
    result: Dict[str, np.ndarray] = {}
    for name in data.dtype.names:
        result[name] = np.asarray(data[name])
    return result


def plot_learning_curves(
    metrics: Dict[str, np.ndarray],
    output_dir: Path,
    window: int = 25,
    title_prefix: str = "",
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    episodes = metrics["episode"]

    figs = [
        ("reward", "Episode reward", "learning_curve_reward.png"),
        ("length", "Episode length", "learning_curve_length.png"),
        ("cliff_hits", "Cliff hits", "learning_curve_cliff_hits.png"),
        ("lure_visits", "Lure visits", "learning_curve_lure_visits.png"),
        ("avg_td_error", "Average TD error", "learning_curve_td_error.png"),
    ]

    for key, ylabel, filename in figs:
        plt.figure(figsize=(9, 4.8))
        plt.plot(episodes, metrics[key], alpha=0.35, label="raw")
        plt.plot(episodes, moving_average(metrics[key], window), linewidth=2.2, label=f"moving avg ({window})")
        plt.xlabel("Episode")
        plt.ylabel(ylabel)
        plt.title(f"{title_prefix}{ylabel}")
        plt.grid(alpha=0.25)
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / filename, dpi=180)
        plt.close()

    plt.figure(figsize=(9, 4.8))
    plt.plot(episodes, metrics["success"], alpha=0.25, label="raw success")
    plt.plot(episodes, moving_average(metrics["success"], window), linewidth=2.2, label=f"success rate ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Success")
    plt.title(f"{title_prefix}Success rate")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "learning_curve_success.png", dpi=180)
    plt.close()


def extract_greedy_policy(q_table: np.ndarray) -> np.ndarray:
    return np.argmax(q_table, axis=1)


def plot_value_heatmap(
    env: RewardMisspecGridworld,
    q_table: np.ndarray,
    output_path: Path,
    title: str = "State-value heatmap (max_a Q)",
) -> None:
    values = q_table.max(axis=1).reshape(env.height, env.width)
    plt.figure(figsize=(1.15 * env.width, 1.05 * env.height))
    plt.imshow(values, interpolation="nearest")
    plt.colorbar(label="max_a Q(s,a)")
    plt.title(title)
    for r in range(env.height):
        for c in range(env.width):
            label = f"{values[r, c]:.1f}"
            plt.text(c, r, label, ha="center", va="center", fontsize=8)
    plt.xticks(range(env.width))
    plt.yticks(range(env.height))
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def plot_policy(
    env: RewardMisspecGridworld,
    q_table: np.ndarray,
    output_path: Path,
    title: str = "Greedy policy",
) -> None:
    policy = extract_greedy_policy(q_table)
    plt.figure(figsize=(1.15 * env.width, 1.05 * env.height))
    ax = plt.gca()
    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_xticks(range(env.width))
    ax.set_yticks(range(env.height))
    ax.grid(True)

    for r in range(env.height):
        for c in range(env.width):
            s = (r, c)
            idx = env.state_to_index(s)
            if s == env.start:
                ax.text(c, r, "S", ha="center", va="center", fontsize=14, fontweight="bold")
            elif s == env.goal:
                ax.text(c, r, "G", ha="center", va="center", fontsize=14, fontweight="bold")
            elif s in env.cliff_cells:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, alpha=0.35))
                ax.text(c, r, "C", ha="center", va="center", fontsize=12, fontweight="bold")
            elif env.reward_mode == "lure" and s in env.lure_cells:
                ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, alpha=0.15))
                ax.text(c, r, f"L\n{ARROW_MAP[int(policy[idx])]}", ha="center", va="center", fontsize=11)
            else:
                ax.text(c, r, ARROW_MAP[int(policy[idx])], ha="center", va="center", fontsize=16)

    ax.set_title(title)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)
    plt.close()


def compare_runs(
    csv_paths: Iterable[Path],
    labels: Iterable[str],
    output_path: Path,
    metric: str = "reward",
    window: int = 25,
    title: str | None = None,
) -> None:
    plt.figure(figsize=(9, 4.8))
    for csv_path, label in zip(csv_paths, labels):
        metrics = load_metrics(csv_path)
        x = metrics["episode"]
        y = moving_average(metrics[metric], window)
        plt.plot(x, y, linewidth=2.1, label=label)
    plt.xlabel("Episode")
    plt.ylabel(metric.replace("_", " ").title())
    plt.title(title or f"Comparison on {metric}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=180)
    plt.close()


def cli() -> None:
    parser = argparse.ArgumentParser(description="Visualization tools for reward misspecification RL project.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    curves_parser = subparsers.add_parser("curves", help="Plot learning curves from one metrics CSV.")
    curves_parser.add_argument("--metrics", type=Path, required=True)
    curves_parser.add_argument("--output-dir", type=Path, required=True)
    curves_parser.add_argument("--window", type=int, default=25)
    curves_parser.add_argument("--title-prefix", type=str, default="")

    compare_parser = subparsers.add_parser("compare", help="Compare several runs on one metric.")
    compare_parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    compare_parser.add_argument("--labels", type=str, nargs="+", required=True)
    compare_parser.add_argument("--output", type=Path, required=True)
    compare_parser.add_argument("--metric", type=str, default="reward")
    compare_parser.add_argument("--window", type=int, default=25)
    compare_parser.add_argument("--title", type=str, default=None)

    args = parser.parse_args()
    if args.command == "curves":
        metrics = load_metrics(args.metrics)
        plot_learning_curves(metrics, args.output_dir, window=args.window, title_prefix=args.title_prefix)
    elif args.command == "compare":
        compare_runs(args.metrics, args.labels, args.output, metric=args.metric, window=args.window, title=args.title)


if __name__ == "__main__":
    cli()
