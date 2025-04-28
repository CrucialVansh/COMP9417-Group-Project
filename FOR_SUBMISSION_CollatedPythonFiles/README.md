# How to run Logistic Regressor's Python Suite

### Data Distribution Shift

### Classifer_FeatureSpaceDiscriminator.py
Trains a logistic regression model on data from the original distribution and the new test distribution. Provides evidence of covariate shifting.

**Run command:** python3 Classifer_FeatureSpaceDiscriminator.py


#### psi.py
Feature selection which prioritises choosing features that are more stable between the two distributions. Note that the chosen features are saved in **chosenFeatures.py** so that other python files do not need to constantly rely on **psi.py** to run.

**Run command:** python3 psi.py

### OriginalXGBoostAlgo_WithPSI.py


Should produce a model file called **parent_model.json**

**Run command:** python3 OriginalXGBoostAlgo_WithPSI.py

### TransferLearningWithPSI.py
Trains a similar XGBoost model on the original training data but using the features selected from the PSI method. Then uses this model to train a second XGBoost that is trained on the second test distribution.

Should produce a model file called **learning_transfer.json**

**Run command** python3 TransferLearningWithPSI.py

#### catBoost.py
Trains 3 CatBoostClassifier models on the training data. One model only uses class weights, another uses lasso for feature importance and the last uses logistic regression ontop.

**Run command:** python3 catBoost.py
