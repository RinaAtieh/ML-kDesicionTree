import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np


def data_split_train_test():
    for num in [4, 10]:
        filename = './Dataset ' + str(num) + '.csv'
        XY = pd.read_csv(filename, header=None)
        if num==10:
            XY[XY.columns[-1]].replace(0.0, 'N',inplace=True)
            XY[XY.columns[-1]].replace(1.0, 'P', inplace=True)
        X = XY.iloc[:, 1:-1].to_numpy()
        Y = XY.iloc[:, -1].to_numpy()

        sss = StratifiedShuffleSplit(n_splits=1, train_size=200, test_size=100)
        train_rows, test_rows = sss.split(X, Y).__next__()
        train_rows = sorted(train_rows)
        test_rows = sorted(test_rows)
        X_train, Y_train = [X[j] for j in train_rows], [Y[j] for j in train_rows]
        X_test, Y_test = [X[j] for j in test_rows], [Y[j] for j in test_rows]

        train_name = 'train_' + str(num) + '.csv'
        test_name = 'test_' + str(num) + '.csv'
        XY_train = pd.DataFrame(X_train)
        XY_train[len(XY_train.columns)] = pd.Series(Y_train)
        XY_train.to_csv(train_name)
        XY_test = pd.DataFrame(X_test)
        XY_test[len(XY_test.columns)] = pd.Series(Y_test)
        XY_test.to_csv(test_name)

#data_split_train_test()
loaded_XY_train_4 = pd.read_csv('train_4.csv')
loaded_X_train_4 = loaded_XY_train_4.iloc[:, 1:].to_numpy()
loaded_Y_train_4 = loaded_XY_train_4.iloc[:, -1].to_numpy()
loaded_XY_test_4 = pd.read_csv('test_4.csv')
loaded_X_test_4 = loaded_XY_test_4.iloc[:, 1:].to_numpy()

loaded_XY_train_10 = pd.read_csv('train_10.csv')
loaded_X_train_10 = loaded_XY_train_10.iloc[:, 1:].to_numpy()
loaded_Y_train_10 = loaded_XY_train_10.iloc[:, -1].to_numpy()
loaded_XY_test_10 = pd.read_csv('test_10.csv')
loaded_X_test_10 = loaded_XY_test_10.iloc[:, 1:].to_numpy()