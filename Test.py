# Author: Julia Skoneczna
import pandas as pd
from DecisionTreeClassifier import DecisionTree
from CrossValidation import CrossValidator

DATA_POR = 'data/student-por.csv'
DATA_MAT = 'data/student-mat.csv'
TARGET = 'Dalc'
TARGET_COL = 26


class TestRunner:

    def __init__(self):
        data_mat = pd.read_csv(DATA_MAT)
        data_por = pd.read_csv(DATA_POR)
        self.data = pd.concat([data_por, data_mat], axis=0)
        self.validator = CrossValidator(self.data, 5)

    def test(self):
        for pair in range(0, len(self.validator.subset_pairs)):
            print("Pair " + str(pair))
            tree = DecisionTree(self.validator.subset_pairs[pair][1].head(n=10), TARGET)
            for index in range(0, len(self.validator.subset_pairs[pair][0])):
                predicted = tree.predict(self.validator.subset_pairs[pair][0].iloc[index, :])
                real = self.validator.subset_pairs[pair][0].iloc[index, TARGET_COL]
                if predicted != real:
                    print("Bad prediction for row " + str(index) + ", predicted " + str(predicted) + ", real value " +
                          str(real))
