from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a small suite of experiments for the poster.")
    parser.add_argument("--episodes", type=int, default=700)
    parser.add_argument("--output-root", type=Path, default=Path("outputs/suite"))
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--algorithms", nargs="+", default=["q_learning", "sarsa"])
    parser.add_argument("--reward-modes", nargs="+", default=["correct", "lure"])
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--lure-bonus", type=float, default=6.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    project_root = Path(__file__).resolve().parent
    train_script = project_root / "train.py"

    for reward_mode in args.reward_modes:
        for algorithm in args.algorithms:
            for seed in args.seeds:
                run_dir = args.output_root / reward_mode / algorithm / f"seed_{seed}"
                cmd = [
                    sys.executable,
                    str(train_script),
                    "--algorithm", algorithm,
                    "--reward-mode", reward_mode,
                    "--episodes", str(args.episodes),
                    "--width", str(args.width),
                    "--height", str(args.height),
                    "--alpha", str(args.alpha),
                    "--gamma", str(args.gamma),
                    "--epsilon", str(args.epsilon),
                    "--epsilon-min", str(args.epsilon_min),
                    "--epsilon-decay", str(args.epsilon_decay),
                    "--lure-bonus", str(args.lure_bonus),
                    "--seed", str(seed),
                    "--output-dir", str(run_dir),
                ]
                print("Running:", " ".join(cmd))
                subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
