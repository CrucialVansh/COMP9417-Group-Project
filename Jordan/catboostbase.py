import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
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
    # Compute class frequencies
    class_counts = np.sum(y_true, axis=0)  # Sum over samples to get counts per class
    class_weights = 1.0 / class_counts
    class_weights /= np.sum(class_weights)  # Normalize weights to sum to 1
    
    # Compute weighted loss
    sample_weights = np.sum(y_true * class_weights, axis=1)  # Get weight for each sample
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))
    
    return loss


X = pd.read_csv('../X_train.csv')
y = pd.read_csv('../y_train.csv')
y = y.values.ravel()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

classes = np.unique(y_train)
weights = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
class_weights = dict(zip(classes, weights))

model = CatBoostClassifier(
    iterations=8000,
    learning_rate=0.01,
    depth=4,
    class_weights=class_weights,
    verbose=100,
    random_state=42,
    task_type='GPU'
)

model.fit(X_train, y_train)
y_pred = model.predict(X_test)

f1_weighted = f1_score(y_test, y_pred, average='weighted')
f1_macro = f1_score(y_test, y_pred, average='macro')
accuracy = accuracy_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"F1 Score (weighted): {f1_weighted:.4f}")
print(f"F1 Score (macro): {f1_macro:.4f}")

y_pred_prob = model.predict_proba(X_test)

y_test_onehot = np.zeros((y_test.shape[0], len(classes)))
y_test_onehot[np.arange(y_test.shape[0]), y_test] = 1

log_loss = weighted_log_loss(y_test_onehot, y_pred_prob)
print(f"Weighted Cross-Entropy Loss: {log_loss:.4f}")