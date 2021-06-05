# Author: Radoslaw Radziukiewicz
import pandas as pd
from CrossValidation import CrossValidator
from RandomForest import RandomForest


class TestRunner:

    def __init__(self, data: pd.DataFrame, target: str):
        self.data = data
        self.target = target

    def _compute_accuracy_with_validator(self, f_size: int, t_size: int, v: CrossValidator) -> float:
        accuracies = []

        for pair in v.subset_pairs:
            forest = RandomForest(f_size, pair[1], self.target, t_size)
            forest.create_forest()
            accuracies.append(AccuracyMetric.measure(forest, pair[0], self.target))

        return sum(accuracies) / len(accuracies)

    def test_cross_validation_split(self, split: list, f_size: int, t_size: int, verbose=True) -> list:
        accuracies = []

        for s_ in split:
            validator = CrossValidator(self.data, s_)
            accuracies.append((split, self._compute_accuracy_with_validator(f_size, t_size, validator)))

            if verbose:
                print("Ended test for split amount: {:d}".format(s_))

        return accuracies

    def test_forest_size(self, split: int, f_size: list, t_size: int, verbose=True) -> list:
        accuracies = []
        validator = CrossValidator(self.data, split)

        for f_ in f_size:
            accuracies.append((f_, self._compute_accuracy_with_validator(f_, t_size, validator)))

            if verbose:
                print("Ended test for forest size: {:d}".format(f_))

        return accuracies

    def test_tree_size_in_forest(self, split: int, f_size: int, t_size: list, verbose=True) -> list:
        accuracies = []
        validator = CrossValidator(self.data, split)

        for t_ in t_size:
            accuracies.append((t_, self._compute_accuracy_with_validator(f_size, t_, validator)))

            if verbose:
                print("Ended test for tree size: {:d}".format(t_))

        return accuracies


class AccuracyMetric:
    @staticmethod
    def measure(classifier, test_dataset: pd.DataFrame, target: str):
        predictions = [classifier.predict(x_) == x_[target] for _, x_ in test_dataset.iterrows()]
        return 100 * sum(predictions) / len(predictions)
