############################
# IA Project 2 - 2019/2020 #
############################

# Joao Parreiro, 89483
# Joao Vieira, 90739

import random
import numpy as np
import matplotlib.pyplot as plt

# LearningAgent to implement
# no knowledge about the environment can be used
# the code should work even with another environment
class LearningAgent:

    def __init__(self, nS, nA):
        self.nS = nS # nS maximum number of states
        self.nA = nA # nA maximum number of action per state
        self.q_table = np.zeros((nS, nA)) # nS lists with nA elements representing each state and its action's q values
        self.visited = np.zeros((nS, nA)) # list to check if each action has been visited
        self.actions_available = [self.nA] * self.nS # list to keep the number of actions available to visit


    # Select one action, used when learning
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontolearn(self, st, aa):
        # set the number of actions that we can choose from
        self.actions_available[st - 1] = len(aa)

        # find the minimum q value in the available actions
        min_val = min(self.visited[st - 1, : len(aa)])

        # find all the available actions with the minimum q value
        min_val_index = []
        for i in range(len(aa)):
            if self.visited[st - 1, i] == min_val:
                min_val_index.append(i)

        # randomly choose between the available minimum q value actions
        m = random.randint(0, len(min_val_index) - 1)
        self.visited[st - 1, min_val_index[m]] += 1

        return min_val_index[m]


    # Select one action, used when evaluating
    # st - is the current state
    # aa - is the set of possible actions
    # for a given state they are always given in the same order
    # returns
    # a - the index to the action in aa
    def selectactiontoexecute(self, st, aa):
        # always choose the action with maximum q value
        actions = []

        for i in range(len(aa)):
            actions.insert(i, self.q_table[st - 1, i])

        a = actions.index(max(actions))

        return a


    # this function is called after every action
    # ost - original state
    # nst - next state
    # a - the index to the action taken
    # r - reward obtained
    def learn(self, ost, nst, a, r):
        # execution of the Q-Learning formula
        y = 0.7
        w = 0.9

        self.q_table[ost - 1, a] = self.q_table[ost - 1, a] + w * (
            r
            + y * np.max(self.q_table[nst - 1, : self.actions_available[nst - 1]])
            - self.q_table[ost - 1, a]
        )

        return