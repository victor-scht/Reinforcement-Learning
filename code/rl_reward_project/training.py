from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Tuple

import numpy as np

from rl_reward_project.agents import QLearningAgent, SARSAAgent
from rl_reward_project.agents.base import AgentConfig, TabularAgent
from rl_reward_project.envs import GridworldConfig, RewardConfig, RewardMisspecGridworld
from rl_reward_project.utils.io_utils import ensure_dir, save_json
from rl_reward_project.utils.metrics import RunLogger
from visualization import (
    plot_grid_policy,
    plot_learning_curves,
    plot_state_visits,
    plot_trajectory,
    plot_value_heatmap,
)


@dataclass
class TrainConfig:
    algorithm: str = "q_learning"
    reward_mode: str = "correct"
    episodes: int = 1000
    alpha: float = 0.2
    gamma: float = 0.95
    epsilon: float = 0.20
    epsilon_min: float = 0.02
    epsilon_decay: float = 0.995
    optimistic_init: float = 0.0
    seed: int = 0
    output_dir: str = "outputs/run"
    moving_average: int = 50


def build_agent(config: TrainConfig, env: RewardMisspecGridworld) -> TabularAgent:
    agent_cfg = AgentConfig(
        n_states=env.n_states,
        n_actions=env.n_actions,
        alpha=config.alpha,
        gamma=config.gamma,
        epsilon=config.epsilon,
        epsilon_min=config.epsilon_min,
        epsilon_decay=config.epsilon_decay,
        optimistic_init=config.optimistic_init,
        seed=config.seed,
    )
    if config.algorithm == "q_learning":
        return QLearningAgent(agent_cfg)
    if config.algorithm == "sarsa":
        return SARSAAgent(agent_cfg)
    raise ValueError(f"Unsupported algorithm: {config.algorithm}")


def train(
    config: TrainConfig,
) -> Tuple[RunLogger, RewardMisspecGridworld, TabularAgent]:
    output_dir = ensure_dir(config.output_dir)
    env = RewardMisspecGridworld(
        config=GridworldConfig(),
        reward_config=RewardConfig(gamma=config.gamma),
        reward_mode=config.reward_mode,
        seed=config.seed,
    )
    agent = build_agent(config, env)
    logger = RunLogger(output_dir=output_dir)
    logger.set_state_shape(env.n_states)

    save_json(
        {
            "train": asdict(config),
            "environment": env.describe(),
        },
        output_dir / "config.json",
    )

    for episode in range(1, config.episodes + 1):
        state = env.reset()
        action = agent.select_action(state)
        done = False
        episode_return = 0.0
        episode_length = 0
        cliff_hits = 0
        risky_visits = 0
        reached_goal = 0
        logger.log_step(state)

        while not done:
            next_state, reward, done, info = env.step(action)
            logger.log_step(next_state)
            episode_return += reward
            episode_length += 1
            cliff_hits += int(bool(info.get("entered_cliff", False)))
            risky_visits += int(bool(info.get("entered_risky", False)))
            reached_goal = int(env.agent_pos == env.goal)

            next_action = None
            if config.algorithm == "sarsa" and not done:
                next_action = agent.select_action(next_state)

            agent.update(state, action, reward, next_state, next_action, done)

            if config.algorithm == "q_learning":
                state = next_state
                action = agent.select_action(state)
            else:
                state = next_state
                if next_action is not None:
                    action = next_action

        logger.log_episode(
            episode=episode,
            episode_return=episode_return,
            episode_length=episode_length,
            success=int(reached_goal),
            reached_goal=int(reached_goal),
            cliff_hits=cliff_hits,
            risky_visits=risky_visits,
            epsilon=agent.epsilon,
        )
        agent.decay_epsilon()

    df = logger.save_metrics()
    logger.summarize()
    np.save(output_dir / "q_table.npy", agent.q_table)

    figures_dir = output_dir / "figures"
    plot_learning_curves(
        df, figures_dir / "learning_curves.png", moving_average=config.moving_average
    )
    plot_grid_policy(env, agent.q_table, figures_dir / "policy.png")
    plot_value_heatmap(env, agent.q_table, figures_dir / "value_heatmap.png")
    plot_state_visits(env, logger.state_visits, figures_dir / "state_visits.png")
    trajectory = env.rollout_greedy(agent.q_table)
    plot_trajectory(env, trajectory, figures_dir / "greedy_trajectory.png")
    return logger, env, agent
