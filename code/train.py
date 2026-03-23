from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from agents.base import AgentConfig
from agents.expected_sarsa import ExpectedSarsaAgent
from agents.q_learning import QLearningAgent
from agents.sarsa import SarsaAgent
from envs.gridworld import RewardMisspecGridworld
from utils.metrics import EpisodeStats, MetricsTracker
from visualization import plot_learning_curves, plot_policy, plot_value_heatmap


AGENTS = {
    "q_learning": QLearningAgent,
    "sarsa": SarsaAgent,
    "expected_sarsa": ExpectedSarsaAgent,
}


def evaluate_greedy_policy(env: RewardMisspecGridworld, q_table: np.ndarray, episodes: int = 20) -> Dict[str, float]:
    rewards = []
    lengths = []
    successes = []
    cliff_hits = []
    for _ in range(episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_len = 0
        ep_cliff = 0
        while not done:
            s_idx = env.state_to_index(state)
            action = int(np.argmax(q_table[s_idx]))
            next_state, reward, done, info = env.step(action)
            ep_reward += reward
            ep_len += 1
            ep_cliff += int(info.get("cliff", False))
            state = next_state
        rewards.append(ep_reward)
        lengths.append(ep_len)
        successes.append(int(state == env.goal))
        cliff_hits.append(ep_cliff)
    return {
        "eval_reward_mean": float(np.mean(rewards)),
        "eval_length_mean": float(np.mean(lengths)),
        "eval_success_rate": float(np.mean(successes)),
        "eval_cliff_hits_mean": float(np.mean(cliff_hits)),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train tabular RL agents on a reward misspecification gridworld.")
    parser.add_argument("--algorithm", choices=sorted(AGENTS.keys()), default="q_learning")
    parser.add_argument("--episodes", type=int, default=600)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--epsilon", type=float, default=0.15)
    parser.add_argument("--epsilon-min", type=float, default=0.02)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=8)
    parser.add_argument("--height", type=int, default=5)
    parser.add_argument("--reward-mode", choices=["correct", "lure", "right_bonus"], default="lure")
    parser.add_argument("--step-reward", type=float, default=-1.0)
    parser.add_argument("--goal-reward", type=float, default=15.0)
    parser.add_argument("--cliff-penalty", type=float, default=-20.0)
    parser.add_argument("--lure-bonus", type=float, default=6.0)
    parser.add_argument("--right-bonus", type=float, default=0.25)
    parser.add_argument("--slip-prob", type=float, default=0.0)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--eval-episodes", type=int, default=30)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/default_run"))
    parser.add_argument("--curve-window", type=int, default=25)
    return parser


def train(args: argparse.Namespace) -> Tuple[RewardMisspecGridworld, object, MetricsTracker]:
    env = RewardMisspecGridworld(
        width=args.width,
        height=args.height,
        reward_mode=args.reward_mode,
        step_reward=args.step_reward,
        goal_reward=args.goal_reward,
        cliff_penalty=args.cliff_penalty,
        lure_bonus=args.lure_bonus,
        right_bonus=args.right_bonus,
        slip_prob=args.slip_prob,
        max_steps=args.max_steps,
        seed=args.seed,
    )

    agent_config = AgentConfig(
        alpha=args.alpha,
        gamma=args.gamma,
        epsilon=args.epsilon,
        epsilon_min=args.epsilon_min,
        epsilon_decay=args.epsilon_decay,
        seed=args.seed,
    )

    agent_cls = AGENTS[args.algorithm]
    agent = agent_cls(args.width * args.height, env.n_actions, agent_config)
    tracker = MetricsTracker()

    for episode in range(1, args.episodes + 1):
        state = env.reset()
        s_idx = env.state_to_index(state)
        action = agent.act(s_idx)
        done = False

        total_reward = 0.0
        steps = 0
        cliff_hits = 0
        lure_visits = 0
        td_errors = []

        while not done:
            next_state, reward, done, info = env.step(action)
            next_idx = env.state_to_index(next_state)
            total_reward += reward
            steps += 1
            cliff_hits += int(info.get("cliff", False))
            lure_visits += int(info.get("lure", False))

            if args.algorithm == "q_learning":
                td = agent.update(s_idx, action, reward, next_idx, done)
                next_action = None
            elif args.algorithm == "sarsa":
                next_action = agent.act(next_idx)
                td = agent.update(s_idx, action, reward, next_idx, next_action, done)
            else:
                td = agent.update(s_idx, action, reward, next_idx, done)
                next_action = agent.act(next_idx)

            td_errors.append(abs(td))
            s_idx = next_idx
            state = next_state
            if next_action is not None:
                action = next_action
            else:
                action = agent.act(s_idx)

        tracker.add(
            EpisodeStats(
                episode=episode,
                reward=total_reward,
                length=steps,
                success=int(state == env.goal),
                cliff_hits=cliff_hits,
                lure_visits=lure_visits,
                avg_td_error=float(np.mean(td_errors) if td_errors else 0.0),
                epsilon=agent.epsilon,
            )
        )
        agent.decay_epsilon()

    return env, agent, tracker


def save_artifacts(args: argparse.Namespace, env: RewardMisspecGridworld, agent, tracker: MetricsTracker) -> None:
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / "q_table.npy", agent.q)
    tracker.save_csv(out / "metrics.csv")

    config_payload = vars(args).copy()
    config_payload["output_dir"] = str(config_payload["output_dir"])
    with (out / "config.json").open("w", encoding="utf-8") as f:
        json.dump(config_payload, f, indent=2)

    eval_env = RewardMisspecGridworld(
        width=args.width,
        height=args.height,
        reward_mode=args.reward_mode,
        step_reward=args.step_reward,
        goal_reward=args.goal_reward,
        cliff_penalty=args.cliff_penalty,
        lure_bonus=args.lure_bonus,
        right_bonus=args.right_bonus,
        slip_prob=0.0,
        max_steps=args.max_steps,
        seed=args.seed + 12345,
    )
    summary = tracker.summary(last_n=min(50, args.episodes))
    summary.update(evaluate_greedy_policy(eval_env, agent.q, episodes=args.eval_episodes))
    tracker.save_summary(out / "summary.json", extra=summary)

    with (out / "policy_ascii.txt").open("w", encoding="utf-8") as f:
        policy = {env.index_to_state(i): int(a) for i, a in enumerate(np.argmax(agent.q, axis=1))}
        f.write(env.render_ascii(policy=policy))
        f.write("\n")

    plot_learning_curves(tracker.to_dict(), out / "figures", window=args.curve_window)
    plot_policy(env, agent.q, out / "figures" / "policy.png", title=f"Greedy policy: {args.algorithm}, {args.reward_mode}")
    plot_value_heatmap(env, agent.q, out / "figures" / "value_heatmap.png")


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    env, agent, tracker = train(args)
    save_artifacts(args, env, agent, tracker)
    print(f"Run completed. Artifacts saved to: {args.output_dir}")
