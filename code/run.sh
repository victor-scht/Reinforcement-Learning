#!/bin/bash

ALGORITHMS=("q_learning" "sarsa")
REWARDS=("correct" "proximity_bad" "potential")
EPISODES=500

for algo in "${ALGORITHMS[@]}"; do
    for reward in "${REWARDS[@]}"; do

        OUT="${algo}_${reward}"

        echo "Running: $algo with $reward"

        python train.py \
            --algorithm "$algo" \
            --reward-mode "$reward" \
            --episodes $EPISODES \
            --alpha 0.2 \
            --gamma 0.95 \
            --epsilon 0.2 \
            --epsilon-decay 0.995 \
            --output-dir "outputs/$OUT"

    done
done
