from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .io_utils import ensure_dir, save_json


@dataclass
class EpisodeRecord:
    episode: int
    episode_return: float
    episode_length: int
    success: int
    reached_goal: int
    cliff_hits: int
    risky_visits: int
    epsilon: float


@dataclass
class RunLogger:
    output_dir: Path
    records: List[EpisodeRecord] = field(default_factory=list)
    state_visits: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.output_dir = ensure_dir(self.output_dir)
        ensure_dir(self.output_dir / "figures")

    def set_state_shape(self, n_states: int) -> None:
        self.state_visits = np.zeros(n_states, dtype=int)

    def log_step(self, state: int) -> None:
        if self.state_visits is None:
            raise RuntimeError("State visit tracker not initialized.")
        self.state_visits[state] += 1

    def log_episode(self, **kwargs: Dict) -> None:
        self.records.append(EpisodeRecord(**kwargs))

    def metrics_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame([r.__dict__ for r in self.records])

    def save_metrics(self) -> pd.DataFrame:
        df = self.metrics_dataframe()
        df.to_csv(self.output_dir / "metrics.csv", index=False)
        if self.state_visits is not None:
            np.save(self.output_dir / "state_visits.npy", self.state_visits)
        return df

    def summarize(self, tail_fraction: float = 0.1) -> Dict:
        df = self.metrics_dataframe()
        if df.empty:
            summary = {}
        else:
            tail_n = max(1, int(len(df) * tail_fraction))
            tail = df.tail(tail_n)
            summary = {
                "episodes": int(len(df)),
                "mean_return": float(df["episode_return"].mean()),
                "mean_length": float(df["episode_length"].mean()),
                "success_rate": float(df["success"].mean()),
                "tail_mean_return": float(tail["episode_return"].mean()),
                "tail_success_rate": float(tail["success"].mean()),
                "tail_mean_length": float(tail["episode_length"].mean()),
                "total_cliff_hits": int(df["cliff_hits"].sum()),
                "total_risky_visits": int(df["risky_visits"].sum()),
            }
        save_json(summary, self.output_dir / "summary.json")
        return summary
