import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, classification_report
import numpy as np
from statistics import mean


def check_conditional_distribution_shift(df1, df2, target_column, features, n_samples_df1=None, seedNum=42):
    """
    Takes in samples from the original training distribution and the new distribution.
    Assigns an extra feature to each sample - if the sample is from the original
    distribution, then this feature will be given a value of 0. Else, this feature
    will be given a value of 1.

    This data is combined and split into a training and validation set. A logsitic
    regression model is trained on the training set and then tries to discriminate 
    between the two distribtutions in the validation set. 

    The primary metric used here is ROC AUC to determien whether the model is able
    to perform better than simple random guessing (i.e. if ROC AUC > 0.5).
    """

    df1_subset = df1.copy()
    df2_subset = df2.copy()
    

    # Chose to undersample from the training data to not oversaturate the 
    # classifier with data from one distribution
    df1_subset = df1_subset.sample(n=202, random_state=seedNum)


    # Create a target variable indicating the dataset origin
    labels_df1 = np.zeros(len(df1_subset))  # 0 for df1
    labels_df2 = np.ones(len(df2_subset))   # 1 for df2
    
    data = pd.concat([df1_subset, df2_subset], ignore_index=True)
    labels = np.concatenate([labels_df1, labels_df2])
    
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

if __name__ == "__main__":
    NUM_DATASET2_LABELED = 202
    features_df1 = pd.read_csv('./data//X_train.csv')

    output_class_df1 = pd.read_csv('./data/y_train.csv')

    features_df2 = pd.read_csv('./data/X_test_2.csv')[ : NUM_DATASET2_LABELED]

    output_class_df2 = pd.read_csv('./data/y_test_2_reduced.csv')

    target_column = 'label'

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