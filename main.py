import pandas as pd
from utility import dataframe_encode, dataframe_decode

DATA_POR = 'data/student-por.csv'
DATA_MAT = 'data/student-mat.csv'


if __name__ == '__main__':
    data_mat = pd.read_csv(DATA_MAT)
    data_por = pd.read_csv(DATA_POR)
    data = pd.concat([data_por, data_mat], axis=0)
    # print(data.info())
    encoders = dataframe_encode(data)
    print(data.head(n=10))
    dataframe_decode(data, encoders)
    print(data.head(n=10))
    # print(data["sex"])
    # encoder = preprocessing.LabelEncoder().fit(data['sex'])

    # print(data.head(n=5))
