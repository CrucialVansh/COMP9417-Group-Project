import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


TOTAL_FEATURES = 300
NUM_SAMPLES = 1000
df_x_train = pd.read_csv("./X_train.csv")
df_y_train = pd.read_csv("./y_train.csv")

X = np.array(df_x_train)
y = np.array(df_y_train)

scaler = MinMaxScaler().fit(X=df_x_train)
X = scaler.fit_transform(X)

normalisationThreshold = 5
samplesWithExtremeFeatures = set()
samplesWithMissingFeatures = set()
for col in range(TOTAL_FEATURES):
    print(f"Starting feature {col}...")
    for row in range(NUM_SAMPLES):

        if X[row][col] == "":
            print(f"Sample {row}; feature {col} is MISSING!!!")
            samplesWithMissingFeatures.add(row)
    
        elif (abs(X[row][col]) >= normalisationThreshold):
            print(f"Sample {row}; feature {col} has normalisation value of {X[row][col]}")
            samplesWithExtremeFeatures.add(row)

print()
print()

print(f"Samples with missing features = {samplesWithMissingFeatures}")
print(f"# of samples with missing features = {len(samplesWithMissingFeatures)}")


print()
print()
print(f"Samples with extreme features = {samplesWithExtremeFeatures}")
print(f"# of samples with extreme features = {len(samplesWithExtremeFeatures)}")