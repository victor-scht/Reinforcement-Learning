#!/bin/bash

# ===============================
# CONFIG
# ===============================

PLOT_DIR="plots"

# Create directory if it doesn't exist
mkdir -p "$PLOT_DIR"

# ===============================
# COMPARISON PLOTS
# ===============================

echo "Generating comparison plots..."

# -------- Q-learning comparison --------
python visualization.py compare \
    --metrics \
    outputs/q_learning_correct/metrics.csv \
    outputs/q_learning_proximity_bad/metrics.csv \
    outputs/q_learning_potential/metrics.csv \
    --labels \
    "Q Correct" \
    "Q Bad" \
    "Q Potential" \
    --metric episode_return \
    --output "$PLOT_DIR/q_learning_return.png"

# -------- SARSA comparison --------
python visualization.py compare \
    --metrics \
    outputs/sarsa_correct/metrics.csv \
    outputs/sarsa_proximity_bad/metrics.csv \
    outputs/sarsa_potential/metrics.csv \
    --labels \
    "SARSA Correct" \
    "SARSA Bad" \
    "SARSA Potential" \
    --metric episode_return \
    --output "$PLOT_DIR/sarsa_return.png"

# -------- SUCCESS RATE comparison (ALL) --------
python visualization.py compare \
    --metrics \
    outputs/q_learning_correct/metrics.csv \
    outputs/q_learning_proximity_bad/metrics.csv \
    outputs/q_learning_potential/metrics.csv \
    outputs/sarsa_correct/metrics.csv \
    outputs/sarsa_proximity_bad/metrics.csv \
    outputs/sarsa_potential/metrics.csv \
    --labels \
    "Q Correct" \
    "Q Bad" \
    "Q Potential" \
    "SARSA Correct" \
    "SARSA Bad" \
    "SARSA Potential" \
    --metric success \
    --output "$PLOT_DIR/all_success.png"

echo "Done. Plots saved in $PLOT_DIR/"
