import xgboost as xgb
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import accuracy_score
import pandas as pd

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import classification_report

X = pd.read_csv("./X_test_2.csv")[202:]

model = xgb.XGBClassifier()
model.load_model("xgboost_model_transfer.json")
y_pred_proba = model.predict_proba(X)

y_pred = model.predict(X)
with np.printoptions(threshold=np.inf):
    print(y_pred_proba)