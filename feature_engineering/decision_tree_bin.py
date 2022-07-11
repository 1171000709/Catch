from sklearn.tree import DecisionTreeClassifier
import pandas as pd
import numpy as np
from sklearn import tree
import matplotlib.pyplot as plt


class DecisionTreeBin(object):
    def __init__(self):
        self.max_leaf_nodes = 4
        self.criterion = 'entropy'
        self.min_samples_leaf = 0.05

    def optimal_binning_boundary(self, x, y):

        # x: arr, y:serise or arr
        boundary = []

        x = np.array(x).reshape(-1)
        # y = y.values

        clf = DecisionTreeClassifier(
            criterion=self.criterion,
            max_leaf_nodes=self.max_leaf_nodes,
            min_samples_leaf=self.min_samples_leaf)

        #
        clf.fit(x.reshape(-1, 1), y)

        #
        # tree.plot_tree(clf)
        # plt.show()

        n_nodes = clf.tree_.node_count
        children_left = clf.tree_.children_left
        children_right = clf.tree_.children_right
        threshold = clf.tree_.threshold

        for i in range(n_nodes):
            #
            if children_left[i] != children_right[i]:
                boundary.append(threshold[i])

        boundary.sort()

        # min_x = x.min()
        # max_x = x.max() + 0.1
        min_x = -np.inf
        max_x = np.inf
        boundary = [min_x] + boundary + [max_x]

        return boundary

    def feature_bin(self, x, y, boundary=None):
        if boundary is None:
            boundary = self.optimal_binning_boundary(x, y)

        label = [i for i in range(self.max_leaf_nodes)]

        # data_ = pd.concat([x, y], axis=1)
        # data_.columns = ['x', 'y']

        binned_feature = pd.cut(x=x, bins=boundary, right=False, labels=label)

        return binned_feature


if __name__ == '__main__':
    df = pd.read_csv('data/hzd_mend.csv')
    # print(df.head(10))

    feature_input = df['']
    target_output = df['rst']

    decision_tree_bin = DecisionTreeBin()

    binned_ = decision_tree_bin.feature_bin(
        feature_input, target_output)
    print('binned_', binned_, type(binned_))

    df[''] = binned_
    # print(df.head(10))









