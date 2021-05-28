import numpy as np
import pandas as pd
from utility import dataframe_encode, dataframe_decode
from DecisionTreeClassifier import DecisionTree

DATA_POR = 'data/student-por.csv'
DATA_MAT = 'data/student-mat.csv'


if __name__ == '__main__':
    data_mat = pd.read_csv(DATA_MAT)
    data_por = pd.read_csv(DATA_POR)
    data = pd.concat([data_por, data_mat], axis=0)
    # print(data.info())
    encoders = dataframe_encode(data)
    # print(data.head(n=10))
    dataframe_decode(data, encoders)
    # print(data.head(n=10))

    prob = data['school'].value_counts() / data['school'].size
    # print(prob)
    # print(- prob * np.log2(prob))
    # print (data['school'].unique())
    prob = data['school']
    # print(data.loc[data['school'] == 'GP']['sex'])

    data_2 = pd.DataFrame({'Outlook': ['S', 'S', 'O', 'R', 'R', 'R', 'O', 'S', 'S', 'R', 'S', 'O', 'O', 'R'],
                           'Play': ['N', 'N', 'Y', 'Y', 'Y', 'N', 'Y', 'N', 'Y', 'Y', 'Y', 'Y', 'Y', 'N'],
                           'Temp': ['H', 'H', 'H', 'M', 'C', 'C', 'C', 'M', 'C', 'M', 'M', 'M', 'H', 'M'],
                           'Hum' : ['H', 'H', 'H', 'H', 'N', 'N', 'N', 'H', 'N', 'N', 'N', 'H', 'N', 'H'],
                           'Wind': ['W', 'S', 'W', 'W', 'W', 'S', 'S', 'W', 'W', 'W', 'S', 'S', 'W', 'S']})

    print(DecisionTree._entropy(data_2['Play']))
    print(DecisionTree._avg_information_entropy(data_2, 'Outlook', 'Play'))
    print(DecisionTree._avg_information_entropy(data_2, 'Temp', 'Play'))
    print(DecisionTree._avg_information_entropy(data_2, 'Hum', 'Play'))
    print(DecisionTree._avg_information_entropy(data_2, 'Wind', 'Play'))

    # print(data["sex"])
    # encoder = preprocessing.LabelEncoder().fit(data['sex'])

    # print(data.head(n=5))
