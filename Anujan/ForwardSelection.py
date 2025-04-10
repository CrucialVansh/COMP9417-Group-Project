import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from sklearn.tree import DecisionTreeClassifier

TOTAL_FEATURES = 300
NUM_SAMPLES = 1000
df_x_train = pd.read_csv("./X_train.csv")
df_y_train = pd.read_csv("./y_train.csv")

X = np.array(df_x_train)
y = np.array(df_y_train)

scaler = MinMaxScaler().fit(X=df_x_train)
X = scaler.fit_transform(X)

X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7)


NUM_FEATURES_TO_SELECT = 50
selectedFeatures = set()
selectedFeatureData = None

selectedValidationData = None
currConfirmedAcc = -1
model = DecisionTreeClassifier(criterion="entropy")
while len(selectedFeatures) < NUM_FEATURES_TO_SELECT:
    print(f"Current length of selected features = {len(selectedFeatures)}")
    bestAccValidation = None
    trainingData = None
    validationData = None
    bestFeature = -1
    for feature in range(TOTAL_FEATURES):
        if feature in selectedFeatures:
            continue
        
        if selectedFeatureData is None:
            trainingData =  X_train[:, feature].reshape(-1, 1)
            validationData = X_val[:, feature].reshape(-1, 1)
        else:
            trainingData = np.concatenate([selectedFeatureData, X_train[:,feature].reshape(-1, 1)], axis=1)
            validationData = np.concatenate([selectedValidationData, X_val[:,feature].reshape(-1, 1)], axis=1)
            
        model.fit(trainingData, y=y_train)
        y_pred = model.predict(validationData)

        acc = accuracy_score(y_val, y_pred)
        if bestAccValidation is None or acc > bestAccValidation:
            bestAccValidation = acc
            bestFeature = feature
    
    print(f"Next most important feature is {bestFeature}; accuracy score was {bestAccValidation}")
    if currConfirmedAcc > bestAccValidation:
        print("Adding a new feature seems to reduce accuracy")
        break
    selectedFeatures.add(bestFeature)

    if selectedFeatureData is None:
        selectedFeatureData =  X_train[:, bestFeature].reshape(-1, 1)
        selectedValidationData = X_val[:, bestFeature].reshape(-1, 1)
    else:
        selectedFeatureData = np.concatenate([selectedFeatureData, X_train[:,bestFeature].reshape(-1, 1)], axis=1)
        selectedValidationData = np.concatenate([selectedValidationData, X_val[:,bestFeature].reshape(-1, 1)], axis=1)


print(f"TOP {NUM_FEATURES_TO_SELECT} are {selectedFeatures}")

