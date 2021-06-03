import pandas as pd
from DecisionTreeClassifier import DecisionTree
from CrossValidation import CrossValidator
from Test import TestRunner

DATA_POR = 'data/student-por.csv'
DATA_MAT = 'data/student-mat.csv'
TARGET = 'Dalc'
TARGET_COL = 26

if __name__ == '__main__':
    test = TestRunner()
    test.test()
    # print(data.head)
    #
    # # Used for previous model verification; Left for possible further bugs.
    # data_2 = pd.DataFrame({'Outlook': ['S', 'S', 'O', 'R', 'R', 'R', 'O', 'S', 'S', 'R', 'S', 'O', 'O', 'R'],
    #                        'Play': ['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N'],
    #                        'Temp': ['H', 'H', 'H', 'M', 'C', 'C', 'C', 'M', 'C', 'M', 'M', 'M', 'H', 'M'],
    #                        'Hum': ['H', 'H', 'H', 'H', 'N', 'N', 'N', 'H', 'N', 'N', 'N', 'H', 'N', 'H'],
    #                        'Wind': ['W', 'S', 'W', 'W', 'W', 'S', 'S', 'W', 'W', 'W', 'S', 'S', 'W', 'S']})
    #
    # print(data.iloc[:100, :10].head(n=10))
    # tree = DecisionTree(data.iloc[:500, :], TARGET)
    #
    # for i in range(0, 500):
    #     if tree.predict(data.iloc[i, :]) != data.iloc[i, TARGET_COL]:
    #         print('WARNING::WRONG CLASSIFICATION AFTER TRAINING AT TRAINING EXAMPLE ' + str(i))
