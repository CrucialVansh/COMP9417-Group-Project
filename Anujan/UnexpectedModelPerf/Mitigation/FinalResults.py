import xgboost as xgb
import numpy as np
import pandas as pd
from chosenFeatures import chosenFeatures

X = pd.read_csv("./X_test_2.csv")[202:]
X = np.array(X)

model = xgb.XGBClassifier()
model.load_model("learning_transfer.json")      # model trained with chosen features and 
                                                # learning transfer

# selecting the features
X = X[: , chosenFeatures]

# prediction with probabilty
y_pred_proba = model.predict_proba(X)

# save the output array to required file
np.savetxt("preds_2.npy", y_pred_proba)