from __future__ import annotations

from dataclasses import dataclass
from typing import Optional
import numpy as np


@dataclass
class AgentConfig:
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_min: float = 0.01
    epsilon_decay: float = 1.0
    seed: Optional[int] = None


class TabularAgent:
    def __init__(self, n_states: int, n_actions: int, config: AgentConfig) -> None:
        self.n_states = n_states
        self.n_actions = n_actions
        self.config = config
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.rng = np.random.default_rng(config.seed)
        self.q = np.zeros((n_states, n_actions), dtype=np.float64)

    def act(self, state_idx: int, greedy: bool = False) -> int:
        if (not greedy) and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.n_actions))
        max_q = np.max(self.q[state_idx])
        greedy_actions = np.flatnonzero(np.isclose(self.q[state_idx], max_q))
        return int(self.rng.choice(greedy_actions))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def greedy_policy(self):
        return np.argmax(self.q, axis=1)
