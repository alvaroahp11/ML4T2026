""""""
"""
Test a learner.  (c) 2015 Tucker Balch

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
"""

import math
import sys
import time

import numpy as np
import matplotlib.pyplot as plt

import LinRegLearner as lrl
import DTLearner as dt
import RTLearner as rt
import BagLearner as bl
import InsaneLearner as il

def gtid():
    """
    :return: The GT ID of the student
    :rtype: int
    """
    return 904197062  # replace with your GT ID number

def evaluate_dt_leaf_sizes(train_x, train_y, test_x, test_y, max_leaf=75, savepath='./images/dt_learner_performance.png'):

    result = np.zeros((2, max_leaf))
    for i in range(1, max_leaf + 1):
        learner = dt.DTLearner(leaf_size=i, verbose=False)
        learner.add_evidence(train_x, train_y)
        inSample = learner.query(train_x)
        # calculate the error
        rmse = math.sqrt(((inSample - train_y) ** 2).sum() / train_y.shape[0])
        result[0, i-1] = rmse
        outSample = learner.query(test_x)
        rmse = math.sqrt(((outSample - test_y) ** 2).sum() / test_y.shape[0])
        result[1, i-1] = rmse

    leaf_sizes = range(1, max_leaf + 1)
    plt.figure()
    plt.plot(leaf_sizes, result[0, :], color='blue', label='In Sample')
    plt.plot(leaf_sizes, result[1, :], color='red', label='Out of Sample')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title('Decision Tree Learner Performance')
    plt.legend()
    plt.grid()
    plt.savefig(savepath)
    plt.close()
    return result

def evaluate_bag_leaf_sizes(train_x, train_y, test_x, test_y, num_bags=20, max_leaf=75, savepath='./images/bag_learner_performance.png'):

    result = np.zeros((2, max_leaf))
    for i in range(1, max_leaf + 1):
        learner = bl.BagLearner(
            learner=dt.DTLearner,
            kwargs={"leaf_size": i},
            bags=num_bags,
            boost=False,
            verbose=False
        )
        learner.add_evidence(train_x, train_y)

        inSample = learner.query(train_x)
        rmse = math.sqrt(((inSample - train_y) ** 2).sum() / train_y.shape[0])
        result[0, i-1] = rmse

        outSample = learner.query(test_x)
        rmse = math.sqrt(((outSample - test_y) ** 2).sum() / test_y.shape[0])
        result[1, i-1] = rmse

    leaf_sizes = range(1, max_leaf + 1)
    plt.figure()
    plt.plot(leaf_sizes, result[0, :], color='blue', label='In Sample')
    plt.plot(leaf_sizes, result[1, :], color='red', label='Out of Sample')
    plt.xlabel('Leaf Size')
    plt.ylabel('RMSE')
    plt.title(f'BagLearner Performance ({num_bags} bags)')
    plt.legend()
    plt.grid()
    plt.savefig(savepath)
    plt.close()
    return result

def depth_from_node(tree, index):
    if tree[index, 0] == -1:
        return 0
    left = index + int(tree[index, 2])
    right = index + int(tree[index, 3])
    return 1 + max(depth_from_node(tree, left), depth_from_node(tree, right))

def r_squared(y_true, y_pred):
    ss_res = ((y_true - y_pred) ** 2).sum()
    ss_tot = ((y_true - y_true.mean()) ** 2).sum()
    return 1 - (ss_res / ss_tot)

def evaluate_dt_vs_rt(train_x, train_y, test_x, test_y, max_leaf=75,
                      savepath_depth='./images/depth_dt_rt.png',
                      savepath_r2='./images/r2_dt_rt.png'):
    dt_depths, rt_depths = [], []
    dt_r2, rt_r2 = [], []

    for i in range(1, max_leaf + 1):
        learner = dt.DTLearner(leaf_size=i, verbose=False)
        learner.add_evidence(train_x, train_y)
        dt_depths.append(depth_from_node(learner.tree, 0))
        dt_r2.append(r_squared(test_y, learner.query(test_x)))

        learner = rt.RTLearner(leaf_size=i, verbose=False)
        learner.add_evidence(train_x, train_y)
        rt_depths.append(depth_from_node(learner.tree, 0))
        rt_r2.append(r_squared(test_y, learner.query(test_x)))

    leaf_sizes = range(1, max_leaf + 1)

    plt.close('all')
    plt.figure()
    plt.plot(leaf_sizes, dt_depths, color='blue', label='DTLearner')
    plt.plot(leaf_sizes, rt_depths, color='red', label='RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('Tree Depth')
    plt.title('DTLearner vs RTLearner: Tree Depth')
    plt.legend()
    plt.grid()
    plt.savefig(savepath_depth)
    plt.close()

    plt.figure()
    plt.plot(leaf_sizes, dt_r2, color='blue', label='DTLearner')
    plt.plot(leaf_sizes, rt_r2, color='red', label='RTLearner')
    plt.xlabel('Leaf Size')
    plt.ylabel('R-Squared')
    plt.title('DTLearner vs RTLearner: R-Squared (Out of Sample)')
    plt.legend()
    plt.grid()
    plt.savefig(savepath_r2)
    plt.close()


if __name__ == "__main__":

    if len(sys.argv) != 2:
        print("Usage: python testlearner.py <filename>")
        sys.exit(1)

    data = np.genfromtxt(sys.argv[1], delimiter=",")

    if np.isnan(data[0]).all():       # first row is a header
        data = data[1:, :]
    if np.isnan(data[:, 0]).all():    # first column is dates
        data = data[:, 1:]

    # compute how much of the data is training and testing
    train_rows = int(0.6 * data.shape[0])

    np.random.seed(gtid())  
    shuffled_indices = np.random.permutation(data.shape[0])
    train_indices = shuffled_indices[:train_rows]
    test_indices = shuffled_indices[train_rows:]

    # separate out training and testing data
    train_x = data[train_indices, 0:-1]
    train_y = data[train_indices, -1]
    test_x = data[test_indices, 0:-1]
    test_y = data[test_indices, -1]
    # run evaluation and save plot

    evaluate_dt_leaf_sizes(train_x, train_y, test_x, test_y)
    evaluate_bag_leaf_sizes(train_x, train_y, test_x, test_y, num_bags=20)
    evaluate_dt_vs_rt(train_x, train_y, test_x, test_y)


