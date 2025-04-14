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


TOTAL_FEATURES = 300

def handleImbalanceViaSmote():
    df_x_train = pd.read_csv("./X_train.csv")
    df_y_train = pd.read_csv("./y_train.csv")

    X = np.array(df_x_train)
    y = np.array(df_y_train)

    s = SMOTE(random_state=12)

    X_res, y_res = s.fit_resample(X=X, y=y)

    NUM_SAMPLES = X_res.shape[0]

    y_res.reshape((-1, 1))

    return (X_res, y_res)


def crossValidation(X_train, y_train):
    scores = cross_val_score(DecisionTreeClassifier(criterion='entropy', random_state=42), X_train, y_train, cv=10, scoring="f1_macro")

    return scores.mean()

def main():

    X, y = handleImbalanceViaSmote()
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.7)

    dt = DecisionTreeClassifier(criterion='entropy', random_state=42)

    bestF1Score = -1
    bestNumFeatures = -1
    for numFeatures in range(1, 301, 30):
        print(f"Trying with {numFeatures} features...")
        X_train_copy = X_train[:,:]
        rfe = RFE(estimator=dt, n_features_to_select=numFeatures, step=0.2, verbose=1)
        rfe.fit(X=X_train_copy, y=y_train)

        features = np.where(rfe.support_)
        print(f"Chosen features = {features}")

        X_train_copy = X_train_copy[:,features[0]]
        f1Score = crossValidation(X_train_copy, y_train)
        print(f"f1 Score for {numFeatures} number of features is {f1Score}")

        if f1Score > bestF1Score:
            bestF1Score = f1Score
            bestNumFeatures = numFeatures
    
    print(f"Best Number of Features = {bestNumFeatures}")

main()