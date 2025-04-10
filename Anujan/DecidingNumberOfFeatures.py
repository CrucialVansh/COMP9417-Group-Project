import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


### Should we set a maximum number of features we are willing to do?


TOTAL_FEATURES = 300
NUM_SAMPLES = 1000
df_x_train = pd.read_csv("./X_train.csv")
df_y_train = pd.read_csv("./y_train.csv")

X = np.array(df_x_train)
y = np.array(df_y_train)

scaler = StandardScaler().fit(X=df_x_train)
X = scaler.fit_transform(X)


X_train, X_val, y_train, y_val = train_test_split(X, y, train_size=0.7)

avgFeatureImportances = [0 for _ in range(TOTAL_FEATURES)]

### getting feature importances
NUM_TRIALS = 50
for trial in range(NUM_TRIALS):
    print(f"Trial {trial + 1}...")
    model = DecisionTreeClassifier(criterion='entropy').fit(X=X_train, y=y_train)
    impAndFeatures = [(model.feature_importances_[i], i) for i in range(TOTAL_FEATURES)]
    for imp, feature in impAndFeatures:
        avgFeatureImportances[feature] += imp / NUM_TRIALS

avgFeatureImportance = [(avgFeatureImportances[i], i) for i in range(TOTAL_FEATURES)]
avgFeatureImportance.sort(reverse=True)
for avgImp, feature in avgFeatureImportance:
    print(f"Feature {feature} with avg importance of {avgImp}")

plt.bar(x=range(TOTAL_FEATURES), height=avgFeatureImportances, edgecolor="black", width=1.0)
plt.xticks(range(TOTAL_FEATURES))
plt.xticks(rotation=90, fontsize=4)

plt.xlabel('Feature Number')
plt.ylabel('Avg Feature Importance')
plt.title(f'Avg Feature Importance for each Feature over {NUM_TRIALS} trials')
plt.show()


# trial_model = DecisionTreeClassifier(criterion="entropy")
# MAX_NUMBER_OF_FEATURES = 300
# bestNumOfFeatures = None
# bestAcc = None
# for trial in range(1, MAX_NUMBER_OF_FEATURES + 1):

#     currFeatures = np.array([feat for imp, feat  in impAndFeatures[:trial]])
#     X_train_trial = X_train[:, currFeatures]
#     X_val_trial = X_val[:, currFeatures]

#     trial_model.fit(X=X_train_trial, y=y_train)
#     y_val_pred = trial_model.predict(X=X_val_trial)

#     acc = accuracy_score(y_val, y_val_pred)

#     if bestAcc is None or acc > bestAcc:
#         print(f"{trial} number of features may be better")
#         bestAcc = acc
#         bestNumOfFeatures = trial
#     else:
#         print(f"{trial} number of features does not seem as good")

# print(f"After all is said and done, the best number of features to use is {bestNumOfFeatures},")
# print(f"specifically {[feat for imp, feat  in impAndFeatures[:bestNumOfFeatures]]}")