from __future__ import annotations

import numpy as np
from .base import TabularAgent, AgentConfig


class ExpectedSarsaAgent(TabularAgent):
    name = "expected_sarsa"

    def __init__(self, n_states: int, n_actions: int, config: AgentConfig) -> None:
        super().__init__(n_states, n_actions, config)

    def expected_q(self, state_idx: int) -> float:
        q_row = self.q[state_idx]
        max_q = np.max(q_row)
        greedy_actions = np.flatnonzero(np.isclose(q_row, max_q))
        probs = np.full(self.n_actions, self.epsilon / self.n_actions, dtype=np.float64)
        probs[greedy_actions] += (1.0 - self.epsilon) / len(greedy_actions)
        return float(np.dot(probs, q_row))

    def update(self, s: int, a: int, r: float, s_next: int, done: bool) -> float:
        target = r if done else r + self.gamma * self.expected_q(s_next)
        td_error = target - self.q[s, a]
        self.q[s, a] += self.alpha * td_error
        return float(td_error)
