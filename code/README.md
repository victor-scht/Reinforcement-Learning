# RL Reward Misspecification Project

This project studies how different reward functions affect learning and behavior in tabular reinforcement learning.

It includes:
- A custom gridworld environment with walls, a start state, a goal state, and an optional risky region.
- Tabular Q-learning and SARSA.
- Three reward modes:
  - `correct`: sparse task-aligned reward.
  - `proximity_bad`: a naive dense shaping reward that gives a bonus for being near the goal at every step.
  - `potential`: potential-based shaping, which preserves the optimal policy in theory.
- Logging of learning curves and state visit statistics.
- Visualization utilities for learning curves, policy arrows, value heatmaps, and trajectory plots.

## Why these reward modes?

The project is inspired by classic and modern work on reward shaping and reward misspecification:

1. **Ng, Harada, and Russell (1999)** showed that only a specific family of shaping rewards—potential-based shaping—guarantees policy invariance. This motivates the `potential` mode.
2. **Amodei et al. (2016)** discuss reward hacking as a concrete safety problem: agents optimize the specified objective, not the intended one.
3. **Lilian Weng (2024)** provides a broad modern survey of reward hacking and specification gaming in RL.

The `proximity_bad` reward is intentionally designed to sound reasonable to a practitioner (“reward the agent for staying close to the goal”), but it can produce poor behavior such as hovering near the goal instead of finishing the task.

## References

- Andrew Y. Ng, Daishi Harada, and Stuart Russell. *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. ICML 1999.
- Dario Amodei et al. *Concrete Problems in AI Safety*. arXiv:1606.06565, 2016.
- Lilian Weng. *Reward Hacking in Reinforcement Learning*, 2024.

## Installation

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick start

Train Q-learning with the task-aligned reward:

```bash
python train.py --algorithm q_learning --reward-mode correct --episodes 1000 --output-dir outputs/q_correct
```

Train SARSA with the naive shaping reward:

```bash
python train.py --algorithm sarsa --reward-mode proximity_bad --episodes 1000 --output-dir outputs/sarsa_bad
```

Compare runs:

```bash
python visualization.py compare \
  --metrics outputs/q_correct/metrics.csv outputs/sarsa_bad/metrics.csv \
  --labels Q_correct SARSA_bad \
  --metric episode_return \
  --output outputs/compare_return.png
```

Run a full experiment suite:

```bash
python run_experiments.py --episodes 1200 --seeds 0 1 2 --output-root outputs/benchmark
```

## Output files

Each run directory contains:
- `metrics.csv`: per-episode learning curves
- `summary.json`: aggregate stats
- `config.json`: exact run configuration
- `q_table.npy`: learned action-value table
- `figures/`: plots and policy visualizations

