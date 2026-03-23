from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import random


Action = int
State = Tuple[int, int]


@dataclass
class StepInfo:
    cliff: bool = False
    lure: bool = False
    slipped: bool = False
    terminated: bool = False


class RewardMisspecGridworld:
    """
    Small episodic gridworld for studying reward misspecification.

    Layout (default width=8, height=5):
      - Start at bottom-left
      - Goal at bottom-right
      - Cliff along the bottom row between start and goal
      - Optional lure cells near the cliff or elsewhere receive positive bonus

    Reward modes:
      - correct: standard cliff-world style shaping
      - lure: extra bonus on entering lure cells, which can attract the agent
      - right_bonus: tiny bonus for moving right, can bias the agent toward risky shortcuts
    """

    ACTIONS = {
        0: (-1, 0),  # up
        1: (0, 1),   # right
        2: (1, 0),   # down
        3: (0, -1),  # left
    }
    ACTION_NAMES = {0: "U", 1: "R", 2: "D", 3: "L"}

    def __init__(
        self,
        width: int = 8,
        height: int = 5,
        reward_mode: str = "correct",
        step_reward: float = -1.0,
        goal_reward: float = 15.0,
        cliff_penalty: float = -20.0,
        lure_bonus: float = 6.0,
        right_bonus: float = 0.25,
        slip_prob: float = 0.0,
        max_steps: int = 200,
        seed: Optional[int] = None,
    ) -> None:
        if width < 4 or height < 3:
            raise ValueError("Grid must be at least 4x3.")
        self.width = width
        self.height = height
        self.reward_mode = reward_mode
        self.step_reward = step_reward
        self.goal_reward = goal_reward
        self.cliff_penalty = cliff_penalty
        self.lure_bonus = lure_bonus
        self.right_bonus = right_bonus
        self.slip_prob = slip_prob
        self.max_steps = max_steps
        self.random = random.Random(seed)

        self.start: State = (height - 1, 0)
        self.goal: State = (height - 1, width - 1)
        self.cliff_cells = {(height - 1, c) for c in range(1, width - 1)}
        # Lure cells are positioned just above part of the cliff.
        self.lure_cells = {
            (height - 2, c) for c in range(max(1, width // 3), width - 2)
        }

        self.current_state: State = self.start
        self.steps_taken: int = 0

    @property
    def n_actions(self) -> int:
        return 4

    @property
    def states(self) -> List[State]:
        return [(r, c) for r in range(self.height) for c in range(self.width)]

    def seed(self, seed: int) -> None:
        self.random.seed(seed)

    def reset(self) -> State:
        self.current_state = self.start
        self.steps_taken = 0
        return self.current_state

    def sample_action(self) -> Action:
        return self.random.randrange(self.n_actions)

    def state_to_index(self, state: State) -> int:
        return state[0] * self.width + state[1]

    def index_to_state(self, idx: int) -> State:
        return idx // self.width, idx % self.width

    def _clip_state(self, state: State) -> State:
        r, c = state
        r = min(max(r, 0), self.height - 1)
        c = min(max(c, 0), self.width - 1)
        return r, c

    def _apply_action(self, state: State, action: Action) -> State:
        dr, dc = self.ACTIONS[action]
        return self._clip_state((state[0] + dr, state[1] + dc))

    def _maybe_slip(self, action: Action) -> Tuple[Action, bool]:
        if self.slip_prob <= 0.0 or self.random.random() >= self.slip_prob:
            return action, False
        alternatives = [a for a in range(self.n_actions) if a != action]
        return self.random.choice(alternatives), True

    def step(self, action: Action):
        self.steps_taken += 1
        applied_action, slipped = self._maybe_slip(action)
        next_state = self._apply_action(self.current_state, applied_action)

        reward = self.step_reward
        terminated = False
        cliff = False
        lure = False

        if next_state in self.cliff_cells:
            reward = self.cliff_penalty
            next_state = self.start
            cliff = True
        elif next_state == self.goal:
            reward = self.goal_reward
            terminated = True
        else:
            if self.reward_mode == "lure" and next_state in self.lure_cells:
                reward += self.lure_bonus
                lure = True
            if self.reward_mode == "right_bonus" and applied_action == 1:
                reward += self.right_bonus

        truncated = self.steps_taken >= self.max_steps and not terminated
        self.current_state = next_state

        info = {
            "cliff": cliff,
            "lure": lure,
            "slipped": slipped,
            "terminated": terminated,
            "truncated": truncated,
        }
        return next_state, reward, terminated or truncated, info

    def transition_preview(self, state: State, action: Action) -> Tuple[State, float, bool]:
        """Deterministic one-step preview used for greedy policy visualization.

        For plotting and policy extraction we ignore stochastic slip and use the nominal action.
        """
        next_state = self._apply_action(state, action)
        reward = self.step_reward
        done = False
        if next_state in self.cliff_cells:
            reward = self.cliff_penalty
            next_state = self.start
        elif next_state == self.goal:
            reward = self.goal_reward
            done = True
        else:
            if self.reward_mode == "lure" and next_state in self.lure_cells:
                reward += self.lure_bonus
            if self.reward_mode == "right_bonus" and action == 1:
                reward += self.right_bonus
        return next_state, reward, done

    def is_terminal(self, state: State) -> bool:
        return state == self.goal

    def render_ascii(self, policy: Optional[Dict[State, Action]] = None) -> str:
        rows = []
        for r in range(self.height):
            chars = []
            for c in range(self.width):
                s = (r, c)
                if s == self.start:
                    chars.append("S")
                elif s == self.goal:
                    chars.append("G")
                elif s in self.cliff_cells:
                    chars.append("C")
                elif s in self.lure_cells and self.reward_mode == "lure":
                    chars.append("L")
                elif policy is not None and s in policy:
                    chars.append(self.ACTION_NAMES[policy[s]])
                else:
                    chars.append(".")
            rows.append(" ".join(chars))
        return "\n".join(rows)
