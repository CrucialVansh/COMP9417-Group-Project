import numpy as np
import pandas as pd


def calculate_psi(expected, actual, bins=10):
    """
    calculate population stability index
    PSI measures distribution shift between data sets
    - how much has the feature changed
    - identify the features most affected by distribution shift

    High PSI values indicate a possible concept drift

    Parameters:
    -----------
    expected : array-like
        Distribution of feature in training/reference data
    actual : array-like
        Distribution of feature in test/new data
    bins : int, optional
        Number of bins for discretization

    Returns:
    --------
    psi : float
        Population Stability Index
    detailed_psi : dict
        Detailed breakdown of PSI calculation
    """
    expected = np.asarray(expected)
    actual = np.asarray(actual)

    exp_hist, bin_edges = np.histogram(expected, bins=bins)
    act_hist, _ = np.histogram(actual, bins=bin_edges)

    # don't divide by 0
    exp_hist = exp_hist.astype(float)
    act_hist = act_hist.astype(float)

    # normalise (percentages)
    exp_pct = exp_hist / exp_hist.sum()
    act_pct = act_hist / act_hist.sum()

    # no log(0)
    exp_pct = np.where(exp_pct == 0, 1e-10, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-10, act_pct)

    psi_values = (act_pct - exp_pct) * np.log(act_pct / exp_pct)
    psi = np.sum(psi_values)

    detailed_psi = {
        'total_psi': psi,
        'bin_psi_values': psi_values,
        'bin_edges': bin_edges,
        'expected_dist': exp_pct,
        'actual_dist': act_pct
    }

    return psi, detailed_psi


def assess_feature_distribution_shift(train_data, test_data):
    """
    Assess distribution shift across all features

    Parameters:
    -----------
    train_data : DataFrame
        Training dataset
    test_data : DataFrame
        Test dataset

    Returns:
    --------
    psi_results : DataFrame
        PSI values for each feature
    """
    # make sure same features in both datasets
    common_features = list(set(train_data.columns) & set(test_data.columns))

    # get PSI for each feature
    psi_results = {}
    for feature in common_features:
        psi_value, details = calculate_psi(
            train_data[feature],
            test_data[feature]
        )
        psi_results[feature] = {
            'psi': psi_value,
            'category': categorize_psi(psi_value)
        }

    return pd.DataFrame.from_dict(psi_results, orient='index')


def categorize_psi(psi):
    """
    Categorize PSI values

    PSI < 0.1: No significant population change
    0.1 <= PSI < 0.25: Moderate population change
    PSI >= 0.25: Significant population change

    https://feature-engine.trainindata.com/en/1.8.x/api_doc/selection/DropHighPSIFeatures.html
    https://coralogix.com/ai-blog/a-practical-introduction-to-population-stability-index-psi/
    > PSI < 0.1: Insignificant change—your model is stable!
    > 0.1 <= PSI < 0.25: Moderate change—consider some adjustments.
    > PSI >= 0.25: Significant change—your model is unstable and needs an update.
    """
    if psi < 0.1:
        return 'No significant change'
    elif psi < 0.2: # get the features close enough to .25. If we use .25 we get 7 features as opposed to 20 for .2
        return 'Moderate change'
    else:
        return 'Significant change'

def visualize_distribution_shift(train_data, test_data, top_n=5):
    """
    Visualize distribution shift for top N most unstable features

    Parameters:
    -----------
    train_data : DataFrame
        Training dataset
    test_data : DataFrame
        Test dataset
    top_n : int, optional
        Number of top features to visualize
    """
    import matplotlib.pyplot as plt
    import seaborn as sns

    psi_results = assess_feature_distribution_shift(train_data, test_data)

    top_features = psi_results.sort_values('psi', ascending=False).head(top_n)

    fig, axes = plt.subplots(1, top_n, figsize=(15, 3))
    fig.suptitle('Distribution Shift for Top Unstable Features')

    for i, (feature, row) in enumerate(top_features.iterrows()):
        sns.histplot(
            x=train_data[feature],
            label='Test Set 1',
            alpha=0.5,
            ax=axes[i]
        )
        sns.histplot(
            x=test_data[feature],
            label='Test Set 2',
            alpha=0.5,
            ax=axes[i]
        )
        axes[i].set_title(f'{feature}\nPSI: {row["psi"]:.2f}')
        axes[i].legend()

    plt.tight_layout()
    plt.show()


# use this to extract features considered stable (PSI < 0.1)
def get_stable_features(X1: pd.DataFrame, X2: pd.DataFrame):
    psi_results = assess_feature_distribution_shift(X1, X2)
    return psi_results.loc[psi_results['category'] == 'No significant change'].index.astype(int).to_numpy()

def demonstrate_psi():
    np.random.seed(42)

    X: pd.DataFrame = pd.read_csv("../../../X_train.csv")
    y: pd.DataFrame = pd.read_csv("../../../y_train.csv")

    X_test_1: pd.DataFrame = pd.read_csv("../../../X_test_1.csv")
    # which will be tested against y_test_1

    # y_test_2 provides labels for first 202 rows in X_test_2
    X_test_2: pd.DataFrame = pd.read_csv("../../../X_test_2.csv")
    y_test_2: pd.DataFrame = pd.read_csv("../../../y_test_2_reduced.csv")

    # Assess the distribution shift
    print(X_test_1.shape, X_test_2.shape)
    psi_results = assess_feature_distribution_shift(X_test_1, X_test_2)
    # psi_results = assess_feature_distribution_shift(X, X_test_1) # nothing of note. No significant changes; model is stable.
    print("PSI Results:\n", psi_results)

    features_with_significant_change = psi_results.loc[psi_results['category'] == 'Significant change'].sort_values('psi', ascending=False)
    # print("PSI values > 0.25 (Most unstable features):")
    # print(features_with_significant_change)
    # print(f"total: {len(features_with_significant_change)}")

    # features_with_moderate_change = psi_results.loc[psi_results['category'] == 'Moderate change'].sort_values('psi', ascending=False)
    # print("0.1 <= PSI values < 0.25 (Moderately unstable features):")
    # print(features_with_moderate_change)
    # print(f"total: {len(features_with_moderate_change)}")

    print(len(get_stable_features(X_test_1, X_test_2)))

    # Visualize top distribution shifts
    visualize_distribution_shift(X_test_1, X_test_2, top_n=5)

# Run demonstration
if __name__ == "__main__":
    demonstrate_psi()