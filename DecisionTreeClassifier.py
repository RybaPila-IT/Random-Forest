import numpy as np


class DecisionTree:

    class TreeNode:

        def __init__(self, data, target):
            # We have achieved only one answer split or we have no features left
            if self._is_one_ans_left(data, target) or data.columns.size == 1:
                self.is_leaf = True
                self.ans = self._get_most_common_label(data, target)
            else:

                split_feature = ""
                split_gain = 0
                data_entropy = self._entropy(data[target])

                for feature in data:
                    if feature != target:
                        feature_gain = data_entropy - self._avg_information_entropy(data, feature, target)
                        split_gain = max(split_gain, feature_gain)
                        split_feature = feature if split_gain == feature_gain else split_feature

                self.is_leaf = False
                self.split_feature = split_feature
                self.sub_trees = {}

                for feature_possibility in data[self.split_feature].unique():
                    self.sub_trees[feature_possibility] = DecisionTree.TreeNode(
                        data[data[self.split_feature] == feature_possibility].drop(self.split_feature, axis=1), target)

        def predict(self, x_):

            if self.is_leaf:
                return self.ans

            return self.sub_trees[x_[self.split_feature]].predict(x_)

        @staticmethod
        def _entropy(data):
            probabilities = data.value_counts() / data.size
            return np.sum(-probabilities * np.log2(probabilities))

        @staticmethod
        def _avg_information_entropy(data, feature, target):
            entropy = 0
            for outcome in data[feature].unique():
                entropy += (data.loc[data[feature] == outcome].shape[0] / data.shape[0]) * \
                           DecisionTree.TreeNode._entropy(data.loc[data[feature] == outcome][target])
            return entropy

        @staticmethod
        def _is_one_ans_left(data, feature):
            return data[feature].unique().size == 1

        @staticmethod
        def _get_most_common_label(data, feature):
            return data[feature].value_counts().idxmax()

    def __init__(self, data, target):
        self._root = self.TreeNode(data, target)

    def predict(self, x_):
        return self._root.predict(x_)
