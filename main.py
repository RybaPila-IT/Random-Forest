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
    print(test.test_forest_size(2, [6], 30))
