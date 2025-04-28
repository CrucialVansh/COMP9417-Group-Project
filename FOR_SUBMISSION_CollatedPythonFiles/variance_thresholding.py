import pandas as pd
from sklearn.feature_selection import VarianceThreshold

def threshold_variance(this_X, threshold=0.1):
    # remove low variance
    variance_selector = VarianceThreshold(threshold=threshold)
    X_reduced = variance_selector.fit_transform(this_X)

    # optionally scale after thresholding
    # X_scaled = StandardScaler().fit_transform(X_reduced)

    print(f"Threshold variance removed {this_X.shape[1] - X_reduced.shape[1]} features with threshold {threshold}")
    return X_reduced

X_train: pd.DataFrame = pd.read_csv("./data/_train.csv")
X_train = X_train.to_numpy()
y_train: pd.DataFrame = pd.read_csv("./data/y_train.csv")
y_train = y_train.to_numpy().reshape(-1)

X_train = threshold_variance(X_train, threshold=0.7)
