import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier

from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss



TOTAL_FEATURES = 300

def getData():
    df_x_train = pd.read_csv("./X_train.csv")
    df_y_train = pd.read_csv("./y_train.csv")

    X = np.array(df_x_train)
    y = np.array(df_y_train)

    return X, y

def featureSelection(X_train, y_train, X_test):
    dt = DecisionTreeClassifier(criterion='entropy')

    rfe = RFE(estimator=dt, n_features_to_select=20, step=0.2, verbose=1)
    rfe.fit(X=X_train, y=y_train)
    features = np.where(rfe.support_)
    X_train = X_train[:,features[0]]
    X_test = X_test[:,features[0]]      

    return X_train, X_test



def main():
    X, y = getData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7, random_state=42)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    X_train, X_test = featureSelection(X_train, y_train, X_test)

    NUM_TRAINING_SAMPLES = X_train.shape[0]
    NUM_CLASSES = 28

    frequencies = {}
    MAX_FREQ = 4479
    for curr_y in y_train:
        frequencies[curr_y] = frequencies.get(curr_y, 0) + 1
    print(frequencies)
    classWeights = {}
    for cl in range(0, 28):
        classWeights[cl] = MAX_FREQ + 1 - frequencies[cl]
        # classWeights[cl] = NUM_TRAINING_SAMPLES / (NUM_CLASSES * frequencies.get(cl))
    print()
    print(classWeights)
    print()
    RF = RandomForestClassifier(random_state=42, class_weight=classWeights, criterion="entropy").fit(X_train, y_train)
    y_pred = RF.predict(X_test)
    print(classification_report(y_test, y_pred))
    print(y_pred)
    print(f"CROSS ENTROPY LOSS = {log_loss(y_test, y_pred)}")




if __name__ == "__main__":
    main()