# test difference between using lasso regularisation selection and not using
import numpy as np
import pandas as pd
from sklearn.feature_selection import RFECV
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelBinarizer
from sklearn.pipeline import Pipeline
from sklearn.metrics import log_loss
from sklearn.utils import compute_class_weight

RANDOM_STATE = 42

# expects nparrays
def lasso_selection(X, y, alpha=0.01, max_iter=1000):
    X_scaled = StandardScaler().fit_transform(X)

    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=RANDOM_STATE)
    lasso.fit(X_scaled, y)
    selected_features = np.where(np.abs(lasso.coef_) > 0)[0]

    return selected_features

def weighted_log_loss(y_true, y_pred, sample_weight=None, classes=None):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    # convert to one hot encoding
    lb = LabelBinarizer()
    if classes is not None:
        lb.fit(classes)
    else:
        lb.fit(y_true)

    # one hot encode y_true
    y_true_encoded = lb.transform(y_true)

    # no log(0)
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)

    if sample_weight is None:
        sample_weight = np.ones(y_true.shape[0])

    loss = -np.sum(y_true_encoded * np.log(y_pred) * sample_weight[:, np.newaxis]) / np.sum(sample_weight)

    return loss

estimator = LogisticRegression(
    penalty="l2",
    solver="lbfgs",
    max_iter=1000,
    class_weight="balanced",  # None for SMOTE, "balanced" for no SMOTE
    random_state=RANDOM_STATE,
)

cv = StratifiedKFold(n_splits=4)

# old feature selection method, always chooses 300 features.
rfecv = RFECV(
    estimator=estimator,
    step=10,
    cv=cv,
    scoring="f1_weighted",  # f1 weighted for multiclass imbalance too (when not using smote)
    n_jobs=-1,
    # random_state=RANDOM_STATE
)

train_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    # ("feature_selection", rfecv), # features will be selected with lasso
    ("classifier", estimator)
])

if __name__ == '__main__':
    X: pd.DataFrame = pd.read_csv("../../X_train.csv")
    y: pd.DataFrame = pd.read_csv("../../y_train.csv")

    X_test_1: pd.DataFrame = pd.read_csv("../../X_test_1.csv")
    # which will be tested against y_test_1

    # y_test_2 provides labels for first 202 rows in X_test_2
    X_test_2: pd.DataFrame = pd.read_csv("../../X_test_2.csv")
    y_test_2: pd.DataFrame = pd.read_csv("../../y_test_2_reduced.csv")

    X = X.to_numpy()
    y = y.to_numpy().reshape(-1)

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)

    selected_features = lasso_selection(X_train, y_train)
    print(f"Selected features: {selected_features}, length: {len(selected_features)}")
    X_train = X_train[:, selected_features]
    X_val = X_val[:, selected_features]

    train_pipeline.fit(X_train, y_train)
    y_pred = train_pipeline.predict(X_val)
    y_proba = train_pipeline.predict_proba(X_val)

    # get sample weights
    class_weights_dict = dict(zip(np.unique(y_train),
                                  compute_class_weight(
                                      class_weight="balanced",
                                      classes=np.unique(y_train),
                                      y=y_train
                                  )))

    # Compute sample weights for validation data
    sample_weights = np.array([class_weights_dict.get(cls, 1.0) for cls in y_val])

    # Compute weighted log loss
    selected_ce_score = weighted_log_loss(
        y_true=y_val,
        y_pred=y_proba,
        sample_weight=sample_weights
    )

    selected_report = classification_report(y_val, y_pred)

    #
    # now use full feature space
    #

    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    train_pipeline.fit(X_train, y_train)
    y_pred = train_pipeline.predict(X_val)
    y_proba = train_pipeline.predict_proba(X_val)
    unselected_ce_score = weighted_log_loss(
        y_true=y_val,
        y_pred=y_proba,
        sample_weight=sample_weights
    )

    unselected_report = classification_report(y_val, y_pred)

    # report
    print("Classification Report (With LASSO selection): ")
    #     accuracy                           0.66      2000
    #    macro avg       0.43      0.49      0.45      2000
    # weighted avg       0.76      0.66      0.70      2000
    print(selected_report)

    print("Classification Report (Without LASSO selection): ")
    #     accuracy                           0.66      2000
    #    macro avg       0.43      0.49      0.44      2000
    # weighted avg       0.75      0.66      0.70      2000
    print(unselected_report)

    print("LASSO selection weighted log loss: ", selected_ce_score) # ~2.86
    print("No selection weighted log loss: ", unselected_ce_score) #  ~3.15
