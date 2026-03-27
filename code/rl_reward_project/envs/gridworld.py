from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Dict, List, Tuple

import numpy as np


class RewardMode(str, Enum):
    CORRECT = "correct"
    PROXIMITY_BAD = "proximity_bad"
    POTENTIAL = "potential"


@dataclass
class GridworldConfig:
    rows: int = 8
    cols: int = 9
    start: Tuple[int, int] = (6, 0)
    goal: Tuple[int, int] = (0, 8)
    max_steps: int = 150
    walls = [
        (1, 3),
        (3, 3),  # removed (2,3)
        (1, 6),
        (3, 6),  # removed (2,6)
        (4, 4),  # removed (4,5)
    ]
    walls += [(0, 5), (1, 5), (2, 5)]

    cliff_cells: List[Tuple[int, int]] = field(
        default_factory=lambda: [(5, c) for c in range(4, 8)]
        + [(6, c) for c in range(4, 8)]
    )
    risky_cells = [
        (1, 7),
        (1, 8),
        (0, 7),
        (2, 5),
        (3, 5),  # NEW
    ]


@dataclass
class RewardConfig:
    step_penalty: float = -1.0
    goal_reward: float = 10.0
    cliff_penalty: float = -25.0
    risk_penalty: float = -2.0
    bad_proximity_scale: float = 7.0
    potential_scale: float = 1.0
    gamma: float = 0.95


class RewardMisspecGridworld:
    ACTIONS: Dict[int, Tuple[int, int]] = {
        0: (-1, 0),  # up
        1: (1, 0),  # down
        2: (0, -1),  # left
        3: (0, 1),  # right
    }
    ACTION_SYMBOLS: Dict[int, str] = {0: "↑", 1: "↓", 2: "←", 3: "→"}

    def __init__(
        self,
        config: GridworldConfig | None = None,
        reward_config: RewardConfig | None = None,
        reward_mode: RewardMode = RewardMode.CORRECT,
        seed: int = 0,
    ) -> None:
        self.config = config or GridworldConfig()
        self.reward_config = reward_config or RewardConfig()
        self.reward_mode = RewardMode(reward_mode)
        self.rng = np.random.default_rng(seed)
        self.rows = self.config.rows
        self.cols = self.config.cols
        self.start = self.config.start
        self.goal = self.config.goal
        self.max_steps = self.config.max_steps
        self.walls = set(self.config.walls)
        self.cliff_cells = set(self.config.cliff_cells)
        self.risky_cells = set(self.config.risky_cells)
        self.n_actions = 4
        self.n_states = self.rows * self.cols
        self.reset()

    def reset(self) -> int:
        self.agent_pos = self.start
        self.steps = 0
        return self.state_to_idx(self.agent_pos)

    def state_to_idx(self, pos: Tuple[int, int]) -> int:
        return pos[0] * self.cols + pos[1]

    def idx_to_state(self, idx: int) -> Tuple[int, int]:
        return divmod(idx, self.cols)

    def in_bounds(self, pos: Tuple[int, int]) -> bool:
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_terminal(self, pos: Tuple[int, int]) -> bool:
        return pos == self.goal

    def manhattan_distance(self, pos: Tuple[int, int]) -> int:
        return abs(pos[0] - self.goal[0]) + abs(pos[1] - self.goal[1])

    def potential(self, pos: Tuple[int, int]) -> float:
        return -float(self.manhattan_distance(pos))

    def transition(
        self, pos: Tuple[int, int], action: int
    ) -> Tuple[Tuple[int, int], float, bool, Dict]:
        if self.rng.random() < 0.01:
            action = self.rng.integers(0, 4)
        dr, dc = self.ACTIONS[action]
        candidate = (pos[0] + dr, pos[1] + dc)
        if (not self.in_bounds(candidate)) or (candidate in self.walls):
            next_pos = pos
        else:
            next_pos = candidate

        reward = self.reward_config.step_penalty
        info: Dict[str, float | bool | str] = {
            "hit_wall": next_pos == pos and candidate != pos,
            "entered_cliff": False,
            "entered_risky": False,
            "reward_mode": self.reward_mode.value,
        }
        done = False

        if next_pos in self.cliff_cells:
            reward += self.reward_config.cliff_penalty
            info["entered_cliff"] = True
            next_pos = self.start
        elif next_pos in self.risky_cells:
            reward += self.reward_config.risk_penalty
            info["entered_risky"] = True

        if next_pos == self.goal:
            reward += self.reward_config.goal_reward
            done = True

        if self.reward_mode == RewardMode.PROXIMITY_BAD and next_pos != self.goal:
            bonus = self.reward_config.bad_proximity_scale / (
                1.0 + self.manhattan_distance(next_pos)
            )
            reward += bonus
            info["shaping_bonus"] = bonus
        elif self.reward_mode == RewardMode.POTENTIAL:
            phi_before = self.potential(pos)
            phi_after = self.potential(next_pos)
            shaped = self.reward_config.potential_scale * (
                self.reward_config.gamma * phi_after - phi_before
            )
            reward += shaped
            info["shaping_bonus"] = shaped
        else:
            info["shaping_bonus"] = 0.0

        return next_pos, reward, done, info

    def step(self, action: int) -> Tuple[int, float, bool, Dict]:
        self.steps += 1
        next_pos, reward, done, info = self.transition(self.agent_pos, action)
        self.agent_pos = next_pos
        if self.steps >= self.max_steps:
            done = True
        return self.state_to_idx(next_pos), reward, done, info

    def rollout_greedy(
        self, q_table: np.ndarray, max_steps: int | None = None
    ) -> List[Tuple[int, int]]:
        pos = self.start
        trajectory = [pos]
        max_steps = max_steps or self.max_steps
        for _ in range(max_steps):
            state = self.state_to_idx(pos)
            action = int(np.argmax(q_table[state]))
            pos, _, done, _ = self.transition(pos, action)
            trajectory.append(pos)
            if done:
                break
        return trajectory

    def describe(self) -> Dict:
        return {
            "grid": asdict(self.config),
            "reward": asdict(self.reward_config),
            "reward_mode": self.reward_mode.value,
        }

    def cell_type(self, pos: Tuple[int, int]) -> str:
        if pos == self.start:
            return "start"
        if pos == self.goal:
            return "goal"
        if pos in self.walls:
            return "wall"
        if pos in self.cliff_cells:
            return "cliff"
        if pos in self.risky_cells:
            return "risky"
        return "empty"

    def grid_labels(self) -> np.ndarray:
        labels = np.full((self.rows, self.cols), " ", dtype=object)
        for r in range(self.rows):
            for c in range(self.cols):
                pos = (r, c)
                cell_type = self.cell_type(pos)
                labels[r, c] = {
                    "start": "S",
                    "goal": "G",
                    "wall": "#",
                    "cliff": "C",
                    "risky": "R",
                    "empty": ".",
                }[cell_type]
        return labels
