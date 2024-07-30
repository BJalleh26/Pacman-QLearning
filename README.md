# Pacman QLearning

## Overview

This project uses Q-Learning to create an intelligent agent that decides what moves Pacman should make in the classic Pacman game.

## Q-Learning Explanation

Q-Learning is a model-free reinforcement learning algorithm that aims to learn the value of an action in a particular state. The "Q" stands for quality, which represents the quality or value of a given action taken in a given state. 

The Q-Learning algorithm works by learning the Q-values for state-action pairs through iterative updates based on the Bellman equation:

\[ Q(s, a) \leftarrow Q(s, a) + \alpha \left[ r + \gamma \max_{a'} Q(s', a') - Q(s, a) \right] \]

where:
- \( s \) is the current state
- \( a \) is the action taken
- \( s' \) is the resulting state after taking action \( a \)
- \( r \) is the reward received after taking action \( a \)
- \( \alpha \) is the learning rate (controls how much new information overrides the old)
- \( \gamma \) is the discount factor (controls the importance of future rewards)

The goal is to find a policy that maximizes the cumulative reward for the agent.

## Installation

To run the Pacman QLearning project, ensure you have Python 3.10 installed. Additionally, you need to install NumPy:

```bash
pip install numpy
