""""""
from cProfile import label
from turtledemo.chaos import plot

from scipy.stats import gstd

"""Assess a betting strategy.

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

Student Name: Tucker Balch (replace with your name)
GT User ID: tb34 (replace with your User ID)
GT ID: 900897987 (replace with your GT ID)
"""

import numpy as np
import matplotlib.pyplot as plt


def author():
    """
    :return: The GT username of the student
    :rtype: str
    """
    return "aperez374"  # replace tb34 with your Georgia Tech username.

def study_group():
    return "aperez374"

def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 904197062  # replace with your GT ID number

def setup_plot():
    plt.figure()
    plt.xlim([0, 300])
    plt.ylim([-256, 100])
    plt.xlabel("Bets")
    plt.ylabel("Win")

def get_spin_result(win_prob):
    """
    Given a win probability between 0 and 1, the function returns whether the probability will result in a win.

    :param win_prob: The probability of winning
    :type win_prob: float
    :return: The result of the spin.
    :rtype: bool
    """

    result = False
    if np.random.random() <= win_prob:
        result = True

    return result

def strategy(win_prob, max_i, i, a, budget = 0, check_budget = False):
    episode_winnings = 0
    j = 1
    bet = 1

    while (episode_winnings < 80 and j < max_i) or (check_budget and budget>=0 and episode_winnings < 80 and j < max_i):

        if get_spin_result(win_prob):
            episode_winnings += bet
            budget += bet
            bet = 1
        else:
            episode_winnings -= bet
            budget -= bet
            bet *= 2

        if check_budget and bet > budget:
            bet = budget

        a[i, j] = episode_winnings
        j+=1


    a[i, j:] = a[i, j - 1]

def exp1_figure1(win_prob):
    setup_plot()
    a = np.zeros((10,1001))

    for i in range(10):
        strategy(win_prob,1000, i, a)
        plt.plot(a[i], label=f"Episode {i+1}")

    plt.title("Experiment 1, Figure 1")
    plt.legend()
    plt.savefig("./images/figure1.png")

    return

def exp1_figure2(win_prob):
    setup_plot()
    a = np.zeros((1000,1001))
    for i in range(1000):
        strategy(win_prob, 1000, i, a)

    gmean = a.mean(axis=0)
    gstd = a.std(axis=0)

    plt.title("Experiment 1, Figure 2")
    plt.plot(gmean, label="Mean")
    plt.plot(gmean+gstd, label="+Standard Deviation")
    plt.plot(gmean-gstd, label="-Standard Deviation")
    plt.legend()
    plt.savefig("./images/figure2.png")

    exp1_figure3(a)

def exp1_figure3(a):

    setup_plot()
    gmedian = np.median(a=a, axis=0)
    gstd = a.std(axis=0)

    plt.title("Experiment 1, Figure 3")
    plt.plot(gmedian, label="Median")
    plt.plot(gmedian+gstd, label="+Standard Deviation")
    plt.plot(gmedian-gstd, label="-Standard Deviation")
    plt.legend()
    plt.savefig("./images/figure3.png")


def exp2_figure4(win_prob):
    setup_plot()
    a = np.zeros((1000, 1001))
    for i in range(1000):
        strategy(win_prob, 1000, i, a, budget= 256, check_budget=True)

    gmean = a.mean(axis=0)
    gstd = a.std(axis=0)

    plt.title("Experiment 2, Figure 4")
    plt.plot(gmean, label="Mean")
    plt.plot(gmean+gstd, label="+Standard Deviation")
    plt.plot(gmean-gstd, label="-Standard Deviation")
    plt.legend()
    plt.savefig("./images/figure4.png")

    exp2_figure5(a)

def exp2_figure5(a):
    setup_plot()
    gmedian = np.median(a=a, axis=0)
    gstd = a.std(axis=0)

    plt.title("Experiment 2, Figure 5")
    plt.plot(gmedian, label="Median")
    plt.plot(gmedian+gstd, label="+Standard Deviation")
    plt.plot(gmedian-gstd, label="-Standard Deviation")
    plt.legend()
    plt.savefig("./images/figure5.png")

def test_code():
    """
    Method to test your code
    """
    win_prob = 18/38  # set appropriately to the probability of a win
    np.random.seed(gtid())  # do this only once

    exp1_figure1(win_prob)
    exp1_figure2(win_prob)
    exp2_figure4(win_prob)

if __name__ == "__main__":
    test_code()
