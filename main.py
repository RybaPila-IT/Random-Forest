import pandas as pd

if __name__ == '__main__':
    data = pd.read_csv('data/student-mat.csv')
    print(data.head(n=5))