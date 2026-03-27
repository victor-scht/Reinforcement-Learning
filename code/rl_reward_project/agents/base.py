from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class AgentConfig:
    n_states: int
    n_actions: int
    alpha: float = 0.1
    gamma: float = 0.99
    epsilon: float = 0.1
    epsilon_min: float = 0.01
    epsilon_decay: float = 1.0
    optimistic_init: float = 0.0
    seed: int = 0


class TabularAgent:
    def __init__(self, config: AgentConfig) -> None:
        self.config = config
        self.n_states = config.n_states
        self.n_actions = config.n_actions
        self.alpha = config.alpha
        self.gamma = config.gamma
        self.epsilon = config.epsilon
        self.epsilon_min = config.epsilon_min
        self.epsilon_decay = config.epsilon_decay
        self.rng = np.random.default_rng(config.seed)
        self.q_table = np.full(
            (self.n_states, self.n_actions), config.optimistic_init, dtype=float
        )

    def select_action(self, state: int, greedy: bool = False) -> int:
        if greedy or self.rng.random() > self.epsilon:
            return self.greedy_action(state)
        return int(self.rng.integers(self.n_actions))

    def greedy_action(self, state: int) -> int:
        row = self.q_table[state]
        max_value = np.max(row)
        candidates = np.flatnonzero(np.isclose(row, max_value))
        return int(self.rng.choice(candidates))

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def state_value(self, state: int) -> float:
        return float(np.max(self.q_table[state]))

    def policy(self) -> np.ndarray:
        return np.argmax(self.q_table, axis=1)

    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: Optional[int],
        done: bool,
    ) -> None:
        raise NotImplementedError
