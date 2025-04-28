import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import f1_score, accuracy_score, classification_report
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier

def weighted_log_loss(y_true, y_pred):
    """
    Compute the weighted cross-entropy (log loss) given true labels and predicted probabilities.
    
    Parameters:
    - y_true: (N, C) One-hot encoded true labels
    - y_pred: (N, C) Predicted probabilities
    
    Returns:
    - Weighted log loss (scalar).
    """
    class_counts = np.sum(y_true, axis=0)
    class_weights = 1.0 / class_counts
    class_weights /= np.sum(class_weights)
    
    sample_weights = np.sum(y_true * class_weights, axis=1)
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))
    
    return loss

def lasso_selection(X, y, alpha=0.01, max_iter=1000):
    X_scaled = StandardScaler().fit_transform(X)
    lasso = Lasso(alpha=alpha, max_iter=max_iter, random_state=42)
    lasso.fit(X_scaled, y)
    selected_features = np.where(lasso.coef_ > 0)[0]
    return selected_features

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


# First Model: CatBoost without Feature Selection
print("Training CatBoost without feature selection...")

model_basic = CatBoostClassifier(
    iterations=8000,
    learning_rate=0.01,
    depth=4,
    class_weights=class_weights,
    random_state=42,
    task_type='GPU'
)

model_basic.fit(X_train, y_train)
y_pred_basic = model_basic.predict(X_test)

# Evaluate
f1_weighted_basic = f1_score(y_test, y_pred_basic, average='weighted')
f1_macro_basic = f1_score(y_test, y_pred_basic, average='macro')
accuracy_basic = accuracy_score(y_test, y_pred_basic)

print("\nCatBoost with class weights Basic Performance:")
print(f"Accuracy: {accuracy_basic:.4f}")
print(f"F1 Score (weighted): {f1_weighted_basic:.4f}")
print(f"F1 Score (macro): {f1_macro_basic:.4f}")

y_pred_prob_basic = model_basic.predict_proba(X_test)
y_test_onehot = np.zeros((y_test.shape[0], len(classes)))
y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1
log_loss_basic = weighted_log_loss(y_test_onehot, y_pred_prob_basic)
print(f"Weighted Cross-Entropy Loss: {log_loss_basic:.4f}")

# Second Model: CatBoost with Lasso Feature Selection
print("\nTraining CatBoost with Lasso feature selection...")

selected_features = lasso_selection(X_train, y_train)
X_train_lasso = X_train.iloc[:, selected_features]
X_test_lasso = X_test.iloc[:, selected_features]

model_lasso = CatBoostClassifier(
    iterations=8000,
    learning_rate=0.01,
    depth=4,
    class_weights=class_weights,
    random_state=42,
    task_type='GPU'
)

model_lasso.fit(X_train_lasso, y_train)
y_pred_lasso = model_lasso.predict(X_test_lasso)

# Evaluate
f1_weighted_lasso = f1_score(y_test, y_pred_lasso, average='weighted')
f1_macro_lasso = f1_score(y_test, y_pred_lasso, average='macro')
accuracy_lasso = accuracy_score(y_test, y_pred_lasso)

print("\nCatBoost with Lasso Feature Selection Performance:")
print(f"Accuracy: {accuracy_lasso:.4f}")
print(f"F1 Score (weighted): {f1_weighted_lasso:.4f}")
print(f"F1 Score (macro): {f1_macro_lasso:.4f}")

y_pred_prob_lasso = model_lasso.predict_proba(X_test_lasso)
log_loss_lasso = weighted_log_loss(y_test_onehot, y_pred_prob_lasso)
print(f"Weighted Cross-Entropy Loss: {log_loss_lasso:.4f}")

# Third Model: CatBoost Embedding + Logistic Regression
print("\nTraining Logistic Regression on CatBoost embeddings...")

# Get leaf indices from CatBoost
train_leaf_indices = model_basic.calc_leaf_indexes(X_train)
test_leaf_indices = model_basic.calc_leaf_indexes(X_test)

# One-hot encode leaf indices
ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=True)
X_train_embedded = ohe.fit_transform(train_leaf_indices)
X_test_embedded = ohe.transform(test_leaf_indices)

# Train Logistic Regression
log_reg = LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced')
log_reg.fit(X_train_embedded, y_train)

y_pred_lr = log_reg.predict(X_test_embedded)
y_pred_prob_lr = log_reg.predict_proba(X_test_embedded)

# Evaluate
f1_weighted_lr = f1_score(y_test, y_pred_lr, average='weighted')
f1_macro_lr = f1_score(y_test, y_pred_lr, average='macro')
accuracy_lr = accuracy_score(y_test, y_pred_lr)

log_loss_lr = weighted_log_loss(y_test_onehot, y_pred_prob_lr)

print("\nLogistic Regression on CatBoost Embedding Performance:")
print(f"Accuracy: {accuracy_lr:.4f}")
print(f"F1 Score (weighted): {f1_weighted_lr:.4f}")
print(f"F1 Score (macro): {f1_macro_lr:.4f}")
print(f"Weighted Cross-Entropy Loss: {log_loss_lr:.4f}")