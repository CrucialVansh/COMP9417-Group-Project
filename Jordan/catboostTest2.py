import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier

def weighted_log_loss(y_true, y_pred, epsilon=1e-15):
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
    with np.errstate(divide='ignore', invalid='ignore'):
        class_weights = np.where(class_counts > 0, 1.0 / class_counts, 0.0)
    class_weights /= np.sum(class_weights)  # Normalize weights to sum to 1
    
    # Compute weighted loss
    sample_weights = np.sum(y_true * class_weights, axis=1)  # Get weight for each sample
    y_pred = np.clip(y_pred, epsilon, 1.0 - epsilon)
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))
    
    return loss


X_train = pd.read_csv('../X_train.csv')
y_train = pd.read_csv('../y_train.csv')
y_train = y_train.values.ravel()

X_test = pd.read_csv('../X_test_2.csv')
X_test = X_test.iloc[:202].reset_index(drop=True)
y_test = pd.read_csv('../y_test_2_reduced.csv')

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