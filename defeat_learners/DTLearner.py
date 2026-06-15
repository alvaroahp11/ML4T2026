""""""
"""
A simple wrapper for linear regression.  (c) 2015 Tucker Balch
Note, this is NOT a correct DTLearner; Replace with your own implementation.
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

import warnings

import numpy as np


class DTLearner(object):
    """
    This is a decision tree learner object that is implemented incorrectly. You should replace this DTLearner with
    your own correct DTLearner from Project 3.

    :param leaf_size: The maximum number of samples to be aggregated at a leaf, defaults to 1.
    :type leaf_size: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
        If verbose = False your code should not generate ANY output. When we test your code, verbose will be False.
    :type verbose: bool
    """

    def __init__(self, leaf_size=1, verbose=False):
        self.leaf_size = leaf_size
        self.verbose = verbose
        self.tree = np.array([])

    def author(self):
        return "aperez374"

    def study_group(self):
        return "aperez374"

    def add_evidence(self, data_x, data_y):
        self.tree = self.build_tree(data_x, data_y)

    def build_tree(self, data_x, data_y):
        if len(data_x) <= self.leaf_size:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        elif np.all(data_y == data_y[0]):
            return np.array([[-1, data_y[0], np.nan, np.nan]])

        correlations = np.zeros(data_x.shape[1])
        for i in range(data_x.shape[1]):
            if np.std(data_x[:, i]) > 0:
                correlations[i] = np.corrcoef(data_x[:, i], data_y)[0, 1]

        correlations = np.nan_to_num(correlations)

        best_feature = np.argmax(np.abs(correlations))
        best_value = np.median(data_x[:, best_feature])

        left_indices = data_x[:, best_feature] <= best_value
        right_indices = data_x[:, best_feature] > best_value

        if left_indices.sum() == 0 or right_indices.sum() == 0:
            return np.array([[-1, np.mean(data_y), np.nan, np.nan]])

        left_tree = self.build_tree(data_x[left_indices], data_y[left_indices])
        right_tree = self.build_tree(data_x[right_indices], data_y[right_indices])

        root = np.array([[best_feature, best_value, 1, left_tree.shape[0] + 1]])
        return np.vstack((root, left_tree, right_tree))

    def query(self, points):
        results = []
        for point in points:
            node_index = 0
            while self.tree[node_index, 0] != -1:
                feature = int(self.tree[node_index, 0])
                value = self.tree[node_index, 1]
                if point[feature] <= value:
                    node_index += int(self.tree[node_index, 2])
                else:
                    node_index += int(self.tree[node_index, 3])
            results.append(self.tree[node_index, 1])
        return np.array(results)


if __name__ == "__main__":
    print("the secret clue is 'zzyzx'")
