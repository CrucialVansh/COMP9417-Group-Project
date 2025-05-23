import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from catboost import CatBoostClassifier

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
    iterations=5000,
    learning_rate=0.01,
    depth=6,
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

# Scores at different depths
# depth 8, 0.7725, 0.7563, 0.4182
# depth 7, 0.7710, 0.7585, 0.42
# depth 6, 0.7605, 0.7548, 0.4338
# depth 5, 0.7405, 0.7439, 0.4227