# Authors: Radosław Radziukiewicz, Julia Skoneczna
import pandas as pd
from Test import TestRunner

DATA_POR = 'data/student-por.csv'
DATA_MAT = 'data/student-mat.csv'
TARGET = 'Dalc'

if __name__ == '__main__':
    data_mat = pd.read_csv(DATA_MAT)
    data_por = pd.read_csv(DATA_POR)
    data_full = pd.concat([data_por, data_mat], axis=0)
    test = TestRunner(data_mat, TARGET)
    # Tree size tests
    # print(test.test_tree_size_in_forest(4, 20, [20, 40, 60]))
    # Forest size tests
    # print(test.test_forest_size(4, [60], 45))
    # Cross validation tests
    print(test.test_cross_validation_split([2], 60, 45))

