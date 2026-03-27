from __future__ import annotations

from .base import TabularAgent


class SARSAAgent(TabularAgent):
    def update(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        next_action: int | None,
        done: bool,
    ) -> None:
        target = reward
        if not done and next_action is not None:
            target += self.gamma * self.q_table[next_state, next_action]
        td_error = target - self.q_table[state, action]
        self.q_table[state, action] += self.alpha * td_error
