from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List
import csv
import json
import numpy as np


@dataclass
class EpisodeStats:
    episode: int
    reward: float
    length: int
    success: int
    cliff_hits: int
    lure_visits: int
    avg_td_error: float
    epsilon: float


class MetricsTracker:
    def __init__(self) -> None:
        self.episodes: List[EpisodeStats] = []

    def add(self, stats: EpisodeStats) -> None:
        self.episodes.append(stats)

    def to_dict(self) -> Dict[str, np.ndarray]:
        keys = EpisodeStats.__annotations__.keys()
        return {k: np.array([getattr(ep, k) for ep in self.episodes]) for k in keys}

    def save_csv(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(EpisodeStats.__annotations__.keys()))
            writer.writeheader()
            for ep in self.episodes:
                writer.writerow(asdict(ep))

    def summary(self, last_n: int = 50) -> Dict[str, float]:
        if not self.episodes:
            return {}
        subset = self.episodes[-last_n:]
        return {
            "episodes": float(len(self.episodes)),
            "mean_reward_last_n": float(np.mean([e.reward for e in subset])),
            "mean_length_last_n": float(np.mean([e.length for e in subset])),
            "success_rate_last_n": float(np.mean([e.success for e in subset])),
            "mean_cliff_hits_last_n": float(np.mean([e.cliff_hits for e in subset])),
            "mean_lure_visits_last_n": float(np.mean([e.lure_visits for e in subset])),
            "final_epsilon": float(self.episodes[-1].epsilon),
        }

    def save_summary(self, path: Path, extra: Dict[str, object] | None = None) -> None:
        payload = self.summary()
        if extra:
            payload.update(extra)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
