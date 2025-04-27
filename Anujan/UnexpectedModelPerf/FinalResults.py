import xgboost as xgb
import numpy as np
import pandas as pd

X = pd.read_csv("./X_test_2.csv")[202:]

model = xgb.XGBClassifier()
model.load_model("xgboost_model_transfer.json")
y_pred_proba = model.predict_proba(X)

y_pred = model.predict(X)
with np.printoptions(threshold=np.inf):
    print(y_pred_proba)