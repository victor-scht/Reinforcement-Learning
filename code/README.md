# Reward Misspecification in Tabular Reinforcement Learning

A lightweight, poster-friendly RL project designed for fast experiments and clean visualizations.

## Project idea
The project studies a simple but interesting question:

**What happens when the reward function is slightly wrong?**

The environment is a small gridworld with a goal and a cliff. In the `lure` reward setting, some cells above the cliff give a positive bonus. This can attract the agent toward behaviors that are good for the reward signal but not ideal for the intended task.

This lets you compare how different tabular control algorithms behave under reward misspecification.

## Included algorithms
- Q-learning
- SARSA
- Expected SARSA

## Folder structure

```text
reward_misspec_rl_project/
├── agents/
│   ├── base.py
│   ├── expected_sarsa.py
│   ├── q_learning.py
│   └── sarsa.py
├── envs/
│   └── gridworld.py
├── utils/
│   └── metrics.py
├── visualization.py
├── train.py
├── run_experiments.py
├── requirements.txt
└── README.md
```

## Install

```bash
pip install -r requirements.txt
```

## Quick start

### 1) Train one run

```bash
python train.py \
  --algorithm q_learning \
  --reward-mode lure \
  --episodes 700 \
  --alpha 0.5 \
  --epsilon 0.15 \
  --epsilon-decay 0.995 \
  --output-dir outputs/q_learning_lure
```

### 2) Train a baseline with the correct reward

```bash
python train.py \
  --algorithm q_learning \
  --reward-mode correct \
  --episodes 700 \
  --output-dir outputs/q_learning_correct
```

### 3) Compare learning curves

```bash
python visualization.py compare \
  --metrics outputs/q_learning_correct/metrics.csv outputs/q_learning_lure/metrics.csv \
  --labels correct lure \
  --metric reward \
  --output outputs/comparisons/q_learning_reward_compare.png
```

## What gets saved
Each training run saves:
- `metrics.csv` with full per-episode learning curves
- `summary.json` with final summary statistics
- `q_table.npy`
- `policy_ascii.txt`
- `figures/learning_curve_*.png`
- `figures/policy.png`
- `figures/value_heatmap.png`

## Suggested experiments for the poster

### Experiment 1: Correct reward vs misspecified reward
- same algorithm
- same hyperparameters
- compare learned policy and learning curves

### Experiment 2: Q-learning vs SARSA under misspecified reward
- compare risky vs conservative behavior
- use `cliff_hits` and `reward` curves

### Experiment 3: Effect of stochasticity
- run with `--slip-prob 0.05` or `0.1`
- see whether the misspecified reward becomes even more problematic

## Main CLI arguments
- `--algorithm {q_learning,sarsa,expected_sarsa}`
- `--episodes`
- `--alpha`
- `--gamma`
- `--epsilon`
- `--epsilon-min`
- `--epsilon-decay`
- `--reward-mode {correct,lure,right_bonus}`
- `--slip-prob`
- `--lure-bonus`
- `--output-dir`

## Poster tip
A strong poster structure would be:
1. Problem statement: reward misspecification
2. Environment design
3. Algorithms
4. Learning curves
5. Learned policy visualizations
6. Main conclusion: RL optimizes the reward you specify, not the task you intended
