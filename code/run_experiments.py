from __future__ import annotations

import argparse
from typing import List


from rl_reward_project.training import TrainConfig, train
from rl_reward_project.utils.io_utils import ensure_dir
from visualization import compare_metrics


DEFAULT_ALGOS = ["q_learning", "sarsa"]
DEFAULT_REWARD_MODES = ["correct", "proximity_bad", "potential"]


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run an experiment suite over algorithms, reward modes, and seeds."
    )
    parser.add_argument("--algorithms", nargs="+", default=DEFAULT_ALGOS)
    parser.add_argument("--reward-modes", nargs="+", default=DEFAULT_REWARD_MODES)
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument("--moving-average", type=int, default=50)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--output-root", type=str, default="outputs/experiment_suite")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    output_root = ensure_dir(args.output_root)
    metric_paths: List[str] = []
    labels: List[str] = []

    for algo in args.algorithms:
        for reward_mode in args.reward_modes:
            for seed in args.seeds:
                run_dir = output_root / f"{algo}__{reward_mode}__seed{seed}"
                config = TrainConfig(
                    algorithm=algo,
                    reward_mode=reward_mode,
                    episodes=args.episodes,
                    alpha=args.alpha,
                    gamma=args.gamma,
                    epsilon=args.epsilon,
                    epsilon_min=args.epsilon_min,
                    epsilon_decay=args.epsilon_decay,
                    optimistic_init=args.optimistic_init,
                    seed=seed,
                    moving_average=args.moving_average,
                    output_dir=str(run_dir),
                )
                train(config)
                metric_paths.append(str(run_dir / "metrics.csv"))
                labels.append(f"{algo}-{reward_mode}-s{seed}")

    compare_dir = ensure_dir(output_root / "comparisons")
    compare_metrics(
        metric_paths,
        labels,
        "episode_return",
        compare_dir / "all_returns.png",
        args.moving_average,
    )
    compare_metrics(
        metric_paths,
        labels,
        "success",
        compare_dir / "all_success.png",
        args.moving_average,
    )
    print(f"Experiment suite finished. Outputs saved to {output_root}")


if __name__ == "__main__":
    main()
