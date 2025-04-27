import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder


def weighted_log_loss(y_true, y_pred):
    """
    Compute the weighted cross-entropy (log loss) given true labels and predicted probabilities.
    
    Parameters:
    - y_true: (N, C) One-hot encoded true labels
    - y_pred: (N, C) Predicted probabilities
    
    Returns:
    - Weighted log loss (scalar).
    """
    # Compute class frequencies
    epsilon = 1e-15
    class_counts = np.sum(y_true, axis=0)  # Sum over samples to get counts per class

    class_counts = np.clip(class_counts, 1, 10000)
    
    class_weights = 1.0 / class_counts
    class_weights /= np.sum(class_weights)  # Normalize weights to sum to 1
    # Compute weighted loss
    sample_weights = np.sum(y_true * class_weights, axis=1)  # Get weight for each sample

    loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))
    
    return loss


# Loading in the original XGBoost model created from the original training data
model_source = xgb.Booster(model_file='xgboost_model.json')

# The below csv files have a few samples from the original training data set and a few
# duplicate samples to ensure every class appears twice in the data set

# This is so a stratified train_test_split can occur
X_target = pd.read_csv("./X_test_2_edited.csv")
y_target = pd.read_csv("./y_test_2_reduced_edited.csv")

X_target = np.array(X_target)
y_target = np.array(y_target)

y_target = y_target.reshape(-1)

X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
    X_target, y_target, test_size=0.2, random_state=123, stratify=y_target
)


# Applying the same XGBoost techniques as the original model
classes = np.unique(y_target_train)

class_weights_array = compute_class_weight(class_weight='balanced', classes=classes, y=y_target_train)

class_weight_dict = dict(zip(classes, class_weights_array))
sample_weights = np.array([class_weight_dict[label] for label in y_target_train])

xgb_2 = xgb.XGBClassifier(
    objective='multi:softprob',
    num_class=len(classes),
    eval_metric='mlogloss',
    use_label_encoder=False,
    max_depth=3,
    reg_alpha=2,
    random_state=42,
    learning_rate=0.05, 
    n_estimators=500,
    min_child_weight=10,
)

# Fitting the new training data to the XGBoost model.
# Notice that we provide the older XGBoost model to the fitting process
# Improves learning performance
xgb_2.fit(X_target_train, y_target_train, xgb_model=model_source, sample_weight=sample_weights)

y_pred = xgb_2.predict_proba(X_target_test)

y_true = np.zeros((y_target_test.shape[0], 28))
y_true[np.arange(y_target_test.shape[0]), y_target_test] = 1

print(f"WEIGHTED LOG LOSS = {weighted_log_loss(y_true, y_pred)}")


