import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeClassifier
from imblearn.over_sampling import SMOTE

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

def handleImbalanceViaSmote(X, y):

    s = SMOTE(random_state=42, k_neighbors=3)

    X_res, y_res = s.fit_resample(X=X, y=y)

    NUM_SAMPLES = X_res.shape[0]

    y_res.reshape((-1, 1))
    print(f"NUM SAMPLES = {NUM_SAMPLES}")

    # counter = {}
    # for y_val in y_res:
    #     counter[y_val] = counter.get(y_val, 0) + 1

    return (X_res, y_res)



def main():
    X, y = getData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8, random_state=42)
    y_train = y_train.reshape(-1)
    y_test = y_test.reshape(-1)

    X_train, X_test = featureSelection(X_train, y_train, X_test)

    X_train, y_train = handleImbalanceViaSmote(X_train, y_train)  

    NUM_TRAINING_SAMPLES = X_train.shape[0]
    NUM_CLASSES = 28

    frequencies = {}
    MAX_FREQ = 4479
    for curr_y in y_train:
        frequencies[curr_y] = frequencies.get(curr_y, 0) + 1

    classWeights = {}
    for cl in range(0, 28):
        classWeights[cl] = MAX_FREQ + 1 - frequencies[cl]
        # classWeights[cl] = NUM_TRAINING_SAMPLES / (NUM_CLASSES * frequencies.get(cl))

    RF = RandomForestClassifier(n_estimators=1000, random_state=42, bootstrap=True, criterion="entropy").fit(X_train, y_train)

    ## Just getting the class frequencies in the training set:
    y_pred = RF.predict(X_train)
    print(classification_report(y_train, y_pred))
    print()

    print()

    y_pred = RF.predict(X_test)
    print(classification_report(y_test, y_pred))


    y_pred = RF.predict_proba(X=X_test)

    print(f"CROSS ENTROPY LOSS = {log_loss(y_test, y_pred)}")

if __name__ == "__main__":
    main()