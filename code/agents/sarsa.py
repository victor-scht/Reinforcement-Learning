from __future__ import annotations

from .base import TabularAgent, AgentConfig


class SarsaAgent(TabularAgent):
    name = "sarsa"

    def __init__(self, n_states: int, n_actions: int, config: AgentConfig) -> None:
        super().__init__(n_states, n_actions, config)

    def update(self, s: int, a: int, r: float, s_next: int, a_next: int, done: bool) -> float:
        target = r if done else r + self.gamma * self.q[s_next, a_next]
        td_error = target - self.q[s, a]
        self.q[s, a] += self.alpha * td_error
        return float(td_error)
