# Author: Julia Skoneczna
import pandas as pd


class CrossValidator:
    """
    Class representing cross validator for splitting data into test and training subsets. Divides given data
    into k chunks of equal size.
    """
    def __init__(self, data: pd.DataFrame, k: int):
        self.k = k
        self.subset_pairs = []

        self.create_subsets(data, k)

    def create_subsets(self, data: pd.DataFrame, k):
        """
        Splits given dataset into k chunks of equal size and puts them in a list of pairs (testing set, training set).

        :param data - pandas DataFrame containing full dataset.
        :param k - number of equal chunks for the dataset to be split into.
        """
        subset_size = len(data) // k
        test_start = 0
        data = data.sample(frac=1).reset_index(drop=True)  # shuffling data

        for _ in range(0, k - 1):
            test_subset = data.iloc[test_start:(test_start + subset_size), :]
            training_subset = data.drop(range(test_start, (test_start + subset_size)))
            self.subset_pairs.append((test_subset, training_subset))
            test_start = test_start + subset_size

        test_subset = data.iloc[test_start:, :]
        training_subset = data.drop(range(test_start, len(data)))
        self.subset_pairs.append((test_subset, training_subset))
