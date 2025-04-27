import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier

# Load data
X = pd.read_csv('../X_train.csv')
y = pd.read_csv('../y_train.csv')
y = y.values.ravel()

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Compute class weights
classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

# Train CatBoost
cat_model = CatBoostClassifier(
    iterations=8000,
    learning_rate=0.01,
    depth=4,
    class_weights=class_weights,
    verbose=100,
    random_state=42,
    task_type='GPU'
)

cat_model.fit(X_train, y_train)

# Get leaf indices from CatBoost
train_leaf_indices = cat_model.calc_leaf_indexes(X_train)
test_leaf_indices = cat_model.calc_leaf_indexes(X_test)

# One-hot encode the leaf indices
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_train_transformed = ohe.fit_transform(train_leaf_indices)
X_test_transformed = ohe.transform(test_leaf_indices)

# Train Logistic Regression on top
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_transformed, y_train)

# Predict
y_pred = log_reg.predict(X_test_transformed)

# Evaluate
f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (weighted): {f1_weighted:.4f}")
print(f"F1 Score (macro): {f1_macro:.4f}")

