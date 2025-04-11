import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


from pprint import pprint

TOTAL_FEATURES = 300
df_x_train = pd.read_csv("./X_train.csv")
df_y_train = pd.read_csv("./y_train.csv")

X = np.array(df_x_train)
y = np.array(df_y_train)

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7)

##### Preprocessing


X_train_feature_selection = np.copy(X_train)
X_val_feature_selection = np.copy(X_val)

numColsInCurrFeatureSpace = TOTAL_FEATURES

keyMap = {}
for i in range(TOTAL_FEATURES):
    keyMap[i] = i

for trial in range(TOTAL_FEATURES):
    print(f"Trial = {trial + 1}")
    models = []
    bestModel = None
    bestMetric = None
    chosenColToDelete = -1
    
    for colToRemove in range(numColsInCurrFeatureSpace):
        print(f"Going to try remove col {colToRemove}")
        X_train_trial = np.delete(X_train_feature_selection, colToRemove, axis=1)
        X_val_trial = np.delete(X_val_feature_selection, colToRemove, axis=1)

        model = DecisionTreeClassifier(criterion="entropy", random_state=4).fit(X=X_train_trial, y=y_train)

        prediction = model.predict(X=X_val_trial)
        acc = accuracy_score(y_val, prediction)
        
        if bestModel is None or bestMetric < acc:
            bestModel = model
            bestMetric = acc
            chosenColToDelete = colToRemove

    print(f"Removing {keyMap[chosenColToDelete]}")

    X_train_feature_selection = np.delete(X_train_feature_selection, chosenColToDelete, axis=1)
    X_val_feature_selection = np.delete(X_val_feature_selection, chosenColToDelete, axis=1)
    for i in range(chosenColToDelete, numColsInCurrFeatureSpace - 2):
        keyMap[i] = keyMap[i + 1]

    del keyMap[numColsInCurrFeatureSpace - 1]

    numColsInCurrFeatureSpace -= 1



