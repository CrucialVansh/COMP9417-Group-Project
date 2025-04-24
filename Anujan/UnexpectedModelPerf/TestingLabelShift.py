import pandas as pd
import numpy as np
from pprint import pprint
from sklearn.preprocessing import StandardScaler


NUM_LABELS = 28
NUM_FEATURES = 300

def printGrid(grid):
    numRows = len(grid)
    numCols = len(grid[0])
    for r in range(numRows):
        for c in range(numCols):
            print(f"{grid[r][c]} ", end="")
        print()

def printRow(arr):
    for x in arr:
        print(f"{x} ", end="")
    print()

def getStatisticalSummary(X, y):
    count = {}
    grid = [[0 for _ in range(NUM_FEATURES)] for _ in range(NUM_LABELS)]

    for i, sample in enumerate(X):
        label = y[i][0]
        count[label] = count.get(label, 0) + 1
        for feature in range(NUM_FEATURES):
            grid[label][feature] += sample[feature]

    for label in range(NUM_LABELS):
        if count.get(label, 0) == 0:
            continue
        freq = count.get(label, 0)
        for feature in range(NUM_FEATURES):
            grid[label][feature] /= freq
    
    return grid

def getNPArrays(fileX, fileY):
    df_x = pd.read_csv(fileX)
    df_y = pd.read_csv(fileY)

    X = np.array(df_x)
    y = np.array(df_y)

    return X, y

if __name__ == "__main__":
    X_train, y_train = getNPArrays("../X_train.csv", "../y_train.csv")
    X_test2, y_test2 = getNPArrays("./X_test_2.csv", "y_test_2_reduced.csv")
    
    NUM_KNOWN_TEST_SAMPLES = 202
    
    X_test2 = X_test2[ : NUM_KNOWN_TEST_SAMPLES, : ]

    scaler = StandardScaler()
    scaler = scaler.fit(X_train)

    printRow(scaler.mean_) 
    # print(scaler.var_)

    scaler2 = StandardScaler()
    scaler2 = scaler2.fit(X_test2)
    print()
    printRow(scaler2.mean_)
    # print(scaler2.var_)

    print()
    printRow(scaler.mean_ - scaler2.mean_)




    # X_train = scaler.transform(X_train)
    # X_test2 = scaler.transform(X_test2)

    # printGrid(getStatisticalSummary(X_train, y_train))

