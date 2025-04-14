import pandas as pd
import numpy as np

from imblearn.over_sampling import SMOTE
from sklearn.feature_selection import RFE

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import cross_val_score

from sklearn.metrics import f1_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

from pprint import pprint


TOTAL_FEATURES = 300

def getData():
    df_x_train = pd.read_csv("./X_train.csv")
    df_y_train = pd.read_csv("./y_train.csv")

    X = np.array(df_x_train)
    y = np.array(df_y_train)

    return X, y

def handleImbalanceViaSmote(X, y):


    print(X.shape)
    print(y.shape)

    s = SMOTE(random_state=42)
    print(s)

    X_res, y_res = s.fit_resample(X=X, y=y)

    NUM_SAMPLES = X_res.shape[0]

    y_res.reshape((-1, 1))
    print(f"NUM SAMPLES = {NUM_SAMPLES}")

    # counter = {}
    # for y_val in y_res:
    #     counter[y_val] = counter.get(y_val, 0) + 1

    return (X_res, y_res)


def crossValidation(X_train, y_train):
    print("Beginning Cross Validation...")
    scores = cross_val_score(DecisionTreeClassifier(criterion='entropy'), X_train, y_train, cv=10, scoring="f1_macro")
    print(f"Average f1 score = {scores.mean()}; std of f1 = {scores.std()}")

def testingSet(model, X_test, y_test):
    print("Beginning testing set...")
    y_pred = model.predict(X_test)
    print(f"Macro F1 Score is {f1_score(y_test, y_pred, average='macro')}")
    print(f"Accuracy score is {accuracy_score(y_test, y_pred)}")

    print(classification_report(y_test, y_pred))


def main():
    ### Handle Class Imbalances using SMOTE ###

    ### After handling imbalance with smote there are 125412
    ### Each class has 4479 samples
    X, y = getData()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)
    X, y = handleImbalanceViaSmote(X_train, y_train)

    ### In X_train, there are 87788 samples; in X_test there are 37,624 samples
    


    ### HANDLE FEATURE SELECTION ###
    dt = DecisionTreeClassifier(criterion='entropy')

    rfe = RFE(estimator=dt, n_features_to_select=210, step=0.2, verbose=1)
    rfe.fit(X=X_train, y=y_train)


    # rfe.support_ gives an boolean array of size NUM_FEATURES where array[i] == True means the i-th feature was selected

    features = np.where(rfe.support_)      
    pprint(features)
    X_train = X_train[:,features[0]]
    X_test = X_test[:,features[0]]

    print(X_train.shape)
    print()
    print(X_test.shape)

    crossValidation(X_train, y_train)

    model = DecisionTreeClassifier(criterion='entropy').fit(X=X_train, y=y_train)

    testingSet(model, X_test, y_test)

main()