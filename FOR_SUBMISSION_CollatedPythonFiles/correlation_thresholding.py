import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def correlation_feature_selection(this_X, threshold=0.8, method='pearson', plot=True, feature_names=None):
    # convert to df
    # feature names do not hold significance at this stage
    if isinstance(this_X, np.ndarray):
        if feature_names is None:
            feature_names = [str(i) for i in range(this_X.shape[1])]
        X_df = pd.DataFrame(this_X, columns=feature_names)
    else:
        X_df = this_X.copy()
        feature_names = X_df.columns.tolist()

    corr_matrix = X_df.corr(method=method).abs()

    if plot:
        plt.figure(figsize=(20, 16))  # Increase figure size
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(corr_matrix, mask=mask, cmap='coolwarm', annot=False,
                    square=True, linewidths=0.5, cbar_kws={"shrink": 0.8})
        plt.xticks(rotation=90, ha='right')  # Rotate x-labels 90 degrees
        plt.tight_layout()
        plt.title('Feature Correlation Matrix')
        plt.show()

    # upper triangle of correlation matrix
    # ignore diagonal comparison
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

    to_drop = set()

    for col in upper_triangle.columns:
        if col in to_drop:
            continue

        highly_correlated = upper_triangle[col][upper_triangle[col] > threshold].index.tolist()

        for correlated_feature in highly_correlated:
            if correlated_feature not in to_drop: # check both features not dropped
                mean_corr_col = upper_triangle[col].mean()
                mean_corr_feature = upper_triangle[correlated_feature].mean()

                # rm the one with greater mean correlation
                if mean_corr_col > mean_corr_feature:
                    to_drop.add(col)
                    break  # exit early
                else:
                    to_drop.add(correlated_feature)

    to_keep = [int(x) for x in feature_names if x not in to_drop]

    print(f"Correlation feature selection removed {len(to_drop)} features with correlations > {threshold}")
    return this_X[:, np.array(to_keep)]

X_train: pd.DataFrame = pd.read_csv("./data/X_train.csv")
X_train = X_train.to_numpy()
y_train: pd.DataFrame = pd.read_csv("./data/y_train.csv")
y_train = y_train.to_numpy().reshape(-1)

X_train = correlation_feature_selection(X_train, plot=True, threshold=0.7)

