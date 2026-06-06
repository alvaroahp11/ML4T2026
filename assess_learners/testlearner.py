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

import numpy as np

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

    
    """
    # create a learner and train it
    learner = lrl.LinRegLearner(verbose=True)  # create a LinRegLearner
    learner.add_evidence(train_x, train_y)  # train it

    # evaluate in sample
    pred_y = learner.query(train_x)  # get the predictions
    rmse = math.sqrt(((train_y - pred_y) ** 2).sum() / train_y.shape[0])
    print()
    print("In sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=train_y)
    print(f"corr: {c[0,1]}")

    # evaluate out of sample
    pred_y = learner.query(test_x)  # get the predictions
    rmse = math.sqrt(((test_y - pred_y) ** 2).sum() / test_y.shape[0])
    print()
    print("Out of sample results")
    print(f"RMSE: {rmse}")
    c = np.corrcoef(pred_y, y=test_y)
    print(f"corr: {c[0,1]}")
    """

    tree1 = dt.DTLearner(verbose=True)
    tree1.add_evidence(train_x, train_y)
    result = tree1.query(test_x)
    rmse = math.sqrt(((test_y - result) ** 2).sum() / test_y.shape[0])

    print("DTLearner results")
    print(f"RMSE: {rmse}")

    



