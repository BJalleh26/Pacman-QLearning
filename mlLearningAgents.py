# mlLearningAgents.py
# parsons/27-mar-2017
#
# A stub for a reinforcement learning agent to work with the Pacman
# piece of the Berkeley AI project:
#
# http://ai.berkeley.edu/reinforcement.html
#
# As required by the licensing agreement for the PacMan AI we have:
#
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).

# This template was originally adapted to KCL by Simon Parsons, but then
# revised and updated to Py3 for the 2022 course by Dylan Cope and Lin Li

from __future__ import absolute_import
from __future__ import print_function

import random

from pacman import Directions, GameState
from pacman_utils.game import Agent


class GameStateFeatures:
    """
    Wrapper class around a game state where you can extract
    useful information for your Q-learning algorithm

    WARNING: We will use this class to test your code, but the functionality
    of this class will not be tested itself
    """

    def __init__(self, state: GameState):
        """
        Args:
            state: A given game state object
        """

        "*** YOUR CODE HERE ***"
        self.ghostPositions = state.getGhostPositions()
        self.pacmanPosition = state.getPacmanPosition()
        self.food = state.getFood()
        self.capsules = state.getCapsules()
        self._win = state.data._win
        self._lose = state.data._lose
        self.walls = state.getWalls()
    
    def getLegalActions(self):
        """
        Returns the legal actions for Pacman.

        Returns:
            legal: a list of legal actions
        """
        if self._win or self._lose: return []
        else:
            legal = []
            (x, y) = self.pacmanPosition
            if (not self.walls[x][y+1]) and ((x, y + 1) not in self.ghostPositions):
                legal.append(Directions.NORTH)
            if (not self.walls[x+1][y]) and ((x + 1, y) not in self.ghostPositions):
                legal.append(Directions.EAST)
            if (not self.walls[x][y-1]) and ((x, y - 1) not in self.ghostPositions):
                legal.append(Directions.SOUTH)
            if (not self.walls[x-1][y]) and ((x - 1, y) not in self.ghostPositions):
                legal.append(Directions.WEST)
            return legal

    def __eq__(self, other):
        """
        Allows two states to be compared.

        Args:
            other: the object being compared with

        Returns:
            False: if either the chosen features do not match or if the object does not exist
            True: if both objects match based on the chosen features
        """
        if other is None:
            return False
        if not self.ghostPositions == other.ghostPositions:
            return False
        if not self.pacmanPosition == other.pacmanPosition:
            return False
        if not self.food == other.food:
            return False
        if not self.capsules == other.capsules:
            return False
        return True

    def __hash__(self):
        """
        Allows states to be keys of dictionaries.

        Returns:
            A hash of the chosen features
        """
        return hash((tuple(self.ghostPositions),
                     self.pacmanPosition,
                     self.food,
                     tuple(self.capsules)))


class QLearnAgent(Agent):

    def __init__(self,
                 alpha: float = 0.5,
                 epsilon: float = 0.4,
                 gamma: float = 0.8,
                 maxAttempts: int = 30,
                 numTraining: int = 10):
        """
        These values are either passed from the command line (using -a alpha=0.5,...)
        or are set to the default values above.

        The given hyperparameters are suggestions and are not necessarily optimal
        so feel free to experiment with them.

        Args:
            alpha: learning rate
            epsilon: exploration rate
            gamma: discount factor
            maxAttempts: How many times to try each action in each state
            numTraining: number of training episodes
        """
        super().__init__()
        self.alpha = lambda count: float(alpha) / count
        self.epsilon = float(epsilon)
        self.gamma = float(gamma)
        self.maxAttempts = int(maxAttempts)
        self.numTraining = int(numTraining)
        # Count the number of games we have played
        self.episodesSoFar = 0
        # Initialise tables
        self.q_values = dict()
        self.frequencies = dict()
        # Initialise previous state variables
        self.prev_action = None
        self.prev_state = None
        self.prev_state_features = None
        self.prev_reward = None

    # Accessor functions for the variable episodesSoFar controlling learning
    def incrementEpisodesSoFar(self):
        self.episodesSoFar += 1

    def getEpisodesSoFar(self):
        return self.episodesSoFar

    def getNumTraining(self):
        return self.numTraining

    # Accessor functions for parameters
    def setEpsilon(self, value: float):
        self.epsilon = value

    def getAlpha(self, value) -> float:
        return self.alpha(value)

    def setAlpha(self, value: float):
        self.alpha = lambda count: value / count

    def getGamma(self) -> float:
        return self.gamma

    def getMaxAttempts(self) -> int:
        return self.maxAttempts

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    @staticmethod
    def computeReward(startState: GameState,
                      endState: GameState) -> float:
        """
        Args:
            startState: A starting state
            endState: A resulting state

        Returns:
            The reward assigned for the given trajectory
        """
        return endState.getScore() - startState.getScore()

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getQValue(self,
                  state: GameStateFeatures,
                  action: Directions) -> float:
        """
        Args:
            state: A given state
            action: Proposed action to take

        Returns:
            Q(state, action)
        """
        self.q_values.setdefault((state, action), 0)
        return self.q_values[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def maxQValue(self, state: GameStateFeatures) -> float:
        """
        Args:
            state: The given state

        Returns:
            The maximum estimated Q-value attainable from the state
        """
        legal = state.getLegalActions()
        if legal == []:
            return 0
        q_values = []
        for move in legal:
            q_value = self.getQValue(state, move)
            q_values.append(q_value)
        return max(q_values)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def learn(self,
              state: GameStateFeatures,
              action: Directions,
              reward: float,
              nextState: GameStateFeatures):
        """
        Performs a Q-learning update.

        Args:
            state: the initial state
            action: the action that was took
            nextState: the resulting state
            reward: the reward received on this trajectory
        """
        max_q = self.maxQValue(nextState)
        q = self.getQValue(state, action)
        count = self.getCount(state, action)
        self.q_values[(state, action)] = q + self.alpha(count) * (reward + self.gamma * max_q - q)

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def updateCount(self,
                    state: GameStateFeatures,
                    action: Directions):
        """
        Updates the stored visitation counts.

        Args:
            state: Starting state
            action: Action taken
        """
        self.frequencies[(state, action)] = self.getCount(state, action) + 1

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getCount(self,
                 state: GameStateFeatures,
                 action: Directions) -> int:
        """
        Args:
            state: Starting state
            action: Action taken

        Returns:
            Number of times that the action has been taken in a given state
        """
        self.frequencies.setdefault((state, action), 0)
        return self.frequencies[(state, action)]

    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def explorationFn(self,
                      utility: float,
                      counts: int) -> float:
        """
        Computes exploration function.
        Returns a more generous value based on the counts
        in order to explore potentially better actions.

        Args:
            utility: expected utility for taking some action a in some given state s
            counts: counts for having taken visited

        Returns:
            The exploration value
        """
        if counts < self.maxAttempts:
            return utility * (counts + 1)
        else:
            return utility

    def qLearningUpdates(self,
                         state: GameState,
                         stateFeatures: GameStateFeatures):
        """
        Perform updates to frequency table and Q-value if a previous
        state exists.

        Args:
            state: the game state
            stateFeatures: the game state features
        """
        if self.prev_state:
            self.updateCount(self.prev_state_features, self.prev_action)
            self.prev_reward = self.computeReward(self.prev_state, state)
            self.learn(self.prev_state_features, self.prev_action, self.prev_reward, stateFeatures)
    
    # WARNING: You will be tested on the functionality of this method
    # DO NOT change the function signature
    def getAction(self, state: GameState) -> Directions:
        """
        Choose an action to take to maximise reward while
        balancing gathering data for learning.
        Utilises both epsilon-greedy and count based exploration.

        Args:
            state: the current state

        Returns:
            The action to take
        """
        # The data we have about the state of the game
        legal = state.getLegalPacmanActions()
        if Directions.STOP in legal:
            legal.remove(Directions.STOP)

        stateFeatures = GameStateFeatures(state)
        self.qLearningUpdates(state, stateFeatures)
        self.prev_state_features = stateFeatures
        self.prev_state = state

        # epsilon-greedy and count based exploration
        if random.random() < self.epsilon:
            self.prev_action =  random.choice(legal)
        else:
            exploration_values = []
            for move in legal:
                q_value = self.getQValue(stateFeatures, move)
                freq = self.getCount(stateFeatures, move)
                exploration_values.append(self.explorationFn(q_value, freq))
            # Obtains the next move according to the first index of the max exploration value
            self.prev_action = legal[exploration_values.index(max(exploration_values))]
        return self.prev_action

    def final(self, state: GameState):
        """
        Handle the end of episodes.
        This is called by the game after a win or a loss.

        Args:
            state: the final game state
        """
        print(f"Game {self.getEpisodesSoFar()} just ended!")

        stateFeatures = GameStateFeatures(state)
        self.qLearningUpdates(state, stateFeatures)
        
        # Resets the class variables for the next episode
        self.prev_reward = None
        self.prev_state = None
        self.prev_state_features = None
        self.prev_action = None

        # Keep track of the number of games played, and set learning
        # parameters to zero when we are done with the pre-set number
        # of training episodes
        self.incrementEpisodesSoFar()
        if self.getEpisodesSoFar() == self.getNumTraining():
            msg = 'Training Done (turning off epsilon and alpha)'
            print('%s\n%s' % (msg, '-' * len(msg)))
            self.setAlpha(0)
            self.setEpsilon(0)
