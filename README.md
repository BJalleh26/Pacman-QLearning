# Pacman QLearning

## Overview

This project uses Q-Learning to create an intelligent agent that decides what moves Pacman should make in the classic Pacman game.

## Q-Learning Explanation

Q-Learning is a model-free reinforcement learning algorithm that aims to learn the value of an action in a particular state. The "Q" stands for quality, which represents the quality or value of a given action taken in a given state. 

The Q-Learning algorithm works by learning the Q-values for state-action pairs through iterative updates based on the Bellman equation:
<br><br>

function Q-LEARNING-AGENT(percept) returns an action <br>
    &emsp;inputs: percept, a percept indicating the current state s' and reward signal r' <br>
    &emsp;persistent: Q, a table of action values indexed by state and action, initially zero <br>
              &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp; N<sub>sa</sub>, a table of frequencies for state-action pairs, initially zero <br>
              &emsp;&emsp;&emsp;&emsp;&emsp;&nbsp;&nbsp;&nbsp;s, a, r, the previous state, action, and reward, initially null <br>
    &emsp;if TERMINAL?(s) then Q[s,None] <- r' <br>
    &emsp;if s is not null then <br>
    &emsp;&emsp;increment N<sub>sa</sub>[s,a] <br>
    &emsp;&emsp;Q[s,a] <- Q[s,a] + $\alpha\$(N<sub>sa</sub>[s,a])(r + $\gamma\$ max<sub>a'</sub> Q[s',a'] - Q[s,a])  <br>
    &emsp;s, a, r <- s', argmax<sub>a'</sub> f(Q[s',a'], N<sub>sa</sub>[s',a']), r' <br>
    &emsp;return a

- Note that $\alpha\$ is a function of the number of visits to s, a <br>
- Ensures convergence

where:
- \( s \) is the current state
- \( a \) is the action taken
- \( s' \) is the resulting state after taking action \( a \)
- \( r \) is the reward received after taking action \( a \)
- \( $\alpha$\) is the learning rate (controls how much new information overrides the old)
- \( $\gamma$ \) is the discount factor (controls the importance of future rewards)

The goal is to find a policy that maximizes the cumulative reward for the agent.

## Installation

To run the Pacman QLearning project, ensure you have Python 3.10 installed. Additionally, you need to install NumPy:

```bash
pip install numpy
```

## Execution
```bash
python pacman.py -p QLearnAgent -l SmallGrid -x 2000 -n 2010
```


## Attribution 

The Pacman AI projects were developed at UC Berkeley (http://ai.berkeley.edu). The core projects and autograders were primarily created by John DeNero (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu). Student side autograding was added by Brad Miller, Nick Hay, and Pieter Abbeel (pabbeel@cs.berkeley.edu).
