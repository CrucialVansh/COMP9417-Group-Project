import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler


# Taken from Archie's Notebook
def useSMOTE(X_train, y_train):
    smote = SMOTE(random_state=42, k_neighbors=3)
    X = X_train.copy()
    y = y_train.copy().reshape(-1)
    X, y = smote.fit_resample(X, y)
    if smote:
        print(f"SMOTE generated an additional {X.shape[0] - X_train.shape[0]} samples.")
        y_resampled_distribution = pd.DataFrame(y).value_counts().reset_index()
        y_resampled_distribution.columns = ["class", "count"]
        print(y_resampled_distribution)
        return X, y

def check_conditional_distribution_shift(df1, df2, target_column, features, n_samples_df1=None):
    """
    For each class cls, looks for distribution shift in P(x | cls). Essentially, takes
    a class cls, and extracts the samples from the original and new data sets which output 
    that class. Assigns a value 0 to the samples that come from the original distribution
    and 1 to the samples that come from the new distribution. A model is trained on a 
    random portion of these samples and validated on another portion. If the model is able
    to distinguish between which samples originate from the original distribution (0) and
    the new distribution (1), this indicates there has been a concept shift of the class.
    """
    results = {}
    unique_classes = sorted(pd.concat([df1[target_column], df2[target_column]]).unique())

    for cls in unique_classes:
        df1_subset = df1[df1[target_column] == cls][features].copy()
        df2_subset = df2[df2[target_column] == cls][features].copy()
        
        
        if len(df1_subset) < 7 or len(df2_subset) < 7:
            print(f"Warning: Class '{cls}' is has insufficient presence in one or both datasets. Skipping.")
            continue

        # if n_samples_df1 is not None and len(df1_subset_original) > n_samples_df1:
        #     df1_subset = df1_subset_original.sample(n=n_samples_df1, random_state=42) # Added random_state for reproducibility
        # else:
        #     df1_subset = df1_subset_original


        # Create a target variable indicating the dataset origin
        labels_df1 = np.zeros(len(df1_subset))  # 0 for df1
        labels_df2 = np.ones(len(df2_subset))   # 1 for df2

        data = pd.concat([df1_subset, df2_subset], ignore_index=True)
        labels = np.concatenate([labels_df1, labels_df2])
        
        # data contains the 300 feature columns + the class column
        # labels contains one column with either a 0 or a 1 for original and new distributions
        # Train a classifier to distinguish between the datasets

        data = StandardScaler().fit_transform(data)
        
        X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.5, stratify=labels, random_state=42)
        X_train, y_train = useSMOTE(X_train, y_train)
        # print()
        # print(X_train)
        # print()
        discriminator = LogisticRegression(solver='lbfgs')  # You can experiment with other classifiers
        
        discriminator.fit(X_train, y_train)
        y_pred_proba = discriminator.predict_proba(X_test)[:, 1]
        auc = roc_auc_score(y_test, y_pred_proba)
        results[cls] = auc

        print(f"\n--- Class: {cls} ---")
        print(f"Dataset Discriminator ROC AUC: {auc:.4f}")
        print("Classification Report:")
        y_pred = discriminator.predict(X_test)
        print(classification_report(y_test, y_pred, target_names=['Dataset 1', 'Dataset 2'], zero_division=0.0))

    return results

# Example Usage (assuming you have your data in CSV files)

# Load the feature data from the first CSV file
NUM_DATASET2_LABELED = 202
features_df1 = pd.read_csv('../X_train.csv')

# Load the output class data from the first CSV file
output_class_df1 = pd.read_csv('../y_train.csv')

# Load the feature data from the second CSV file
features_df2 = pd.read_csv('./X_test_2.csv')[ : NUM_DATASET2_LABELED]

# Load the output class data from the second CSV file
output_class_df2 = pd.read_csv('./y_test_2_reduced.csv')

# Assuming the output class column is named 'output_class' and is in both output_class_df1 and output_class_df2
# You might need to adjust this depending on your CSV file structure
target_column = 'label'

# Combine feature and output class data into single DataFrames
train_df = pd.concat([features_df1, output_class_df1], axis=1)
test_df = pd.concat([features_df2, output_class_df2], axis=1)

# Assuming your features are named 'feature1' through 'feature5'
# You should replace this with the actual names of your feature columns
# features = ['feature1', 'feature2', 'feature3', 'feature4', 'feature5']
features = [str(i) for i in range(300)]

shift_results = check_conditional_distribution_shift(train_df, test_df, target_column, features, n_samples_df1=20)

print("\nOverall Distribution Shift Analysis (P(x | y)):")
for cls, auc in shift_results.items():
    if auc > 0.7 or auc < 0.3:
        print(f"Class '{cls}': Potential significant shift (AUC = {auc:.4f})")
    elif 0.6 <= auc <= 0.7 or 0.3 <= auc <= 0.4:
        print(f"Class '{cls}': Possible moderate shift (AUC = {auc:.4f})")
    else:
        print(f"Class '{cls}': No strong evidence of shift (AUC = {auc:.4f})")