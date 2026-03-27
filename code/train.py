from __future__ import annotations

import argparse

from rl_reward_project.training import TrainConfig, train


def make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train a tabular RL agent on the reward misspecification gridworld."
    )
    parser.add_argument(
        "--algorithm", choices=["q_learning", "sarsa"], default="q_learning"
    )
    parser.add_argument(
        "--reward-mode",
        choices=["correct", "proximity_bad", "potential"],
        default="correct",
    )
    parser.add_argument("--episodes", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.2)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--epsilon", type=float, default=0.20)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--optimistic-init", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--moving-average", type=int, default=50)
    parser.add_argument("--output-dir", type=str, default="outputs/run")
    return parser


def main() -> None:
    args = make_parser().parse_args()
    config = TrainConfig(
        algorithm=args.algorithm,
        reward_mode=args.reward_mode,
        episodes=args.episodes,
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        optimistic_init=args.optimistic_init,
        seed=args.seed,
        moving_average=args.moving_average,
        output_dir=args.output_dir,
    )
    logger, _, _ = train(config)
    summary = logger.summarize()
    print("Training finished.")
    print(summary)


if __name__ == "__main__":
    main()
