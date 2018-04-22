import pandas as pd
from sklearn.model_selection import train_test_split
import os


def main():
    filename = "data/train.csv"
    filename = os.path.abspath(os.path.join(os.pardir, filename))
    data = pd.read_csv(filename)
    Y = data.is_duplicate
    X = data.drop(columns=['is_duplicate', 'id', 'qid1', 'qid2'])
    X, X_test, Y, Y_test = train_test_split(X, Y, test_size=0.2)
    X_train, X_valid, Y_train, Y_valid = train_test_split(X, Y, test_size=0.2)
    del X, Y


if __name__ == '__main__':
    main()