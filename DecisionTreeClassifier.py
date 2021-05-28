import numpy as np
import pandas as pd
from abc import ABC

class DecisionTree:

    class __Node(ABC):
        def decide(self, data):
            pass

    class __InnerNode(__Node):
        pass

    class __Leaf(__Node):
        pass

    @staticmethod
    def _entropy(data):
        probabilities = data.value_counts() / data.size
        return np.sum(-probabilities * np.log2(probabilities))

    @staticmethod
    def _avg_information_entropy(data, feature, target):
        entropy = 0
        for outcome in data[feature].unique():
            entropy += (data.loc[data[feature] == outcome].shape[0] / data.shape[0]) * \
                       DecisionTree._entropy(data.loc[data[feature] == outcome][target])
        return entropy
