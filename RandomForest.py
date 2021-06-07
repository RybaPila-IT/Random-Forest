# Author: Julia Skoneczna
from pandas import DataFrame
import numpy as np
import math

from DecisionTreeClassifier import DecisionTree


class RandomForest:
    """
    Creates random forest with a specified number of trees.
    """
    def __init__(self, number_of_trees: int, data: DataFrame, target_column_name: str, tree_size: int):
        self.number_of_trees = number_of_trees
        self.data = data
        self.target_column_name = target_column_name
        self.forest = []
        self.tree_size = tree_size

    def create_forest(self):
        for tree_index in range(0, self.number_of_trees):
            random_subset = []
            random_subset = self.pick_random_subset(random_subset)
            random_attributes = []
            random_attributes = self.pick_random_attributes()
            tree_data = self.data.iloc[random_subset, random_attributes]
            tree = DecisionTree(tree_data, self.target_column_name)
            self.forest.append(tree)

    def pick_random_subset(self, random_subset: list) -> list:
        for row_index in range(0, self.tree_size):
            random_subset.append(np.random.randint(0, len(self.data)))
        return random_subset

    def pick_random_attributes(self) -> list:
        attributes = [col for col in self.data.columns if col != self.target_column_name]
        for _ in range(0, len(self.data.columns) - math.floor(math.sqrt(len(self.data.columns)))):
            attribute_index = np.random.randint(0, len(attributes))
            attributes.pop(attribute_index)

        attributes_to_take = self.data.drop(attributes, axis=1)
        return [self.data.columns.get_loc(col) for col in attributes_to_take.columns]

    def predict(self, data_to_classify):
        """
        Classifies data by checking which answer is the most common in the entire forest.

        :param data_to_classify - data to be classified.
        :return final_answer - most common result in the entire forest.
        """
        answer_occurrences = {}
        max_occurrences = -1
        final_answer = 0
        for tree in self.forest:
            current_answer = tree.predict(data_to_classify)
            if current_answer in answer_occurrences:
                answer_occurrences[current_answer] = answer_occurrences[current_answer] + 1
            else:
                answer_occurrences[current_answer] = 1
            if max_occurrences < answer_occurrences[current_answer]:
                max_occurrences = answer_occurrences[current_answer]
                final_answer = current_answer

        return final_answer
