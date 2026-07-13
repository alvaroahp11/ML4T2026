""""""
"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Alvaro Andres Henriquez Perez
GT User ID: aperez374
GT ID: 904197062
"""

import random as rand

import numpy as np


class QLearner(object):
    """
    This is a Q learner object.

    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available..
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2,
        gamma=0.9,
        rar=0.5,
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        """
        Constructor method
        """
        self.verbose = verbose
        self.num_actions = num_actions
        self.num_states = num_states
        self.s = 0
        self.a = 0
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.q_table = np.zeros((num_states, num_actions))
        self.model_t = {}
        self.model_r = {}

    def querysetstate(self, s):
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """
        self.s = s

        if self.rar > rand.random():
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[s])

        if self.verbose:
            print(f"s = {s}, a = {action}")
        
        self.a = action

        return action

    def query(self, s_prime, r):
        """
        Update the Q table and return an action
        
        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """
        self.q_table[self.s, self.a] += self.alpha * (r + self.gamma * np.max(self.q_table[s_prime]) - self.q_table[self.s, self.a])

        if self.dyna > 0:
            self.model_t[(self.s, self.a)] = s_prime
            self.model_r[(self.s, self.a)] = r

            for _ in range(self.dyna):
                s_rand, a_rand = rand.choice(list(self.model_t.keys()))
                s_prime_rand = self.model_t[(s_rand, a_rand)]
                r_rand = self.model_r[(s_rand, a_rand)]

                self.q_table[s_rand, a_rand] += self.alpha * (r_rand + self.gamma * np.max(self.q_table[s_prime_rand]) - self.q_table[s_rand, a_rand])

        if self.rar > rand.random():
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.q_table[s_prime])

        if self.verbose:
            print(f"s = {s_prime}, a = {action}, r = {r}")

        self.rar *= self.radr

        self.s = s_prime
        self.a = action

        return action
    
    def author(self):
        return "aperez374"  

    def study_group(self):
        return "aperez374"


if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
