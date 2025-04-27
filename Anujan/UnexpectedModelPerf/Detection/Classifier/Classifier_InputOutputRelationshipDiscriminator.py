import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import StandardScaler
from statistics import mean

# Taken from Archie's Notebook
# def useSMOTE(X_train, y_train):
#     smote = SMOTE(random_state=42, k_neighbors=3)
#     X = X_train.copy()
#     y = y_train.copy().reshape(-1)
#     X, y = smote.fit_resample(X, y)
#     if smote:
#         return X, y

def check_conditional_distribution_shift(df1, df2, target_column, features, n_samples_df1=None, seedNum=42):
    """
    For each class cls, looks for distribution shift in P(x | cls). Essentially, takes
    a class cls, and extracts the samples from the original and new data sets which output 
    that class. Assigns a value 0 to the samples that come from the original distribution
    and 1 to the samples that come from the new distribution. A model is trained on a 
    random portion of these samples and validated on another portion. If the model is able
    to distinguish between which samples originate from the original distribution (0) and
    the new distribution (1), this indicates there has been a concept shift of the class.
    """

    df1_subset = df1.copy()
    df2_subset = df2.copy()
    

    df1_subset = df1_subset.sample(n=202, random_state=seedNum)


    # Create a target variable indicating the dataset origin
    labels_df1 = np.zeros(len(df1_subset))  # 0 for df1
    labels_df2 = np.ones(len(df2_subset))   # 1 for df2

    data = pd.concat([df1_subset, df2_subset], ignore_index=True)
    labels = np.concatenate([labels_df1, labels_df2])
    
    # data contains the 300 feature columns + the class column
    # labels contains one column with either a 0 or a 1 for original and new distributions
    # Train a classifier to distinguish between the datasets

    data = StandardScaler().fit_transform(data)
    
    X_train, X_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, stratify=labels, random_state=seedNum)

    discriminator = LogisticRegression(solver='lbfgs', max_iter=1000)
    
    discriminator.fit(X_train, y_train)
    y_pred_proba = discriminator.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_pred_proba)
    results =  auc

    print(f"Dataset Discriminator ROC AUC: {auc:.4f}")
    print("Classification Report:")
    y_pred = discriminator.predict(X_test)
    print(classification_report(y_test, y_pred, target_names=['Dataset 1', 'Dataset 2'], zero_division=0.0))

    return results

# Example Usage (assuming you have your data in CSV files)

if __name__ == "__main__":
    #  Load the feature data from the first CSV file
    NUM_DATASET2_LABELED = 202
    features_df1 = pd.read_csv('../X_train.csv')

    # Load the output class data from the first CSV file
    output_class_df1 = pd.read_csv('../y_train.csv')

    # Load the feature data from the second CSV file
    features_df2 = pd.read_csv('./X_test_2.csv')[ : NUM_DATASET2_LABELED]

    # Load the output class data from the second CSV file
    output_class_df2 = pd.read_csv('./y_test_2_reduced.csv')

    target_column = 'label'

    # Combine feature and output class data into single DataFrames
    train_df = pd.concat([features_df1, output_class_df1], axis=1)
    test_df = pd.concat([features_df2, output_class_df2], axis=1)
    features = [str(i) for i in range(300)]

    results = []
    for trial in range(1000):
        print(f"Trial {trial + 1}")
        shift_results = check_conditional_distribution_shift(train_df, test_df, target_column, features, n_samples_df1=20, seedNum=trial)
        results.append(shift_results)


    print("\nOverall Distribution Shift Analysis (P(x | y)):")
    shift_results = mean(results)
    if shift_results > 0.7 or shift_results < 0.3:
        print(f"Potential significant shift (shift_results = {shift_results:.4f})")
    elif 0.6 <= shift_results <= 0.7 or 0.3 <= shift_results <= 0.4:
        print(f"Possible moderate shift (shift_results = {shift_results:.4f})")
    else:
        print(f"No strong evidence of shift (shift_results = {shift_results:.4f})")