import numpy as np
from MMD import mmd_rbf
import pandas as pd

def mmd_permutation_test(X, Y, n_permutations=1000):
    observed_mmd = mmd_rbf(X, Y)
    print(f"Observed mmd = {observed_mmd}")
    combined_data = np.vstack((X, Y))
    print(combined_data.shape)
    n_x = len(X)
    count_extreme = 0
    for i in range(n_permutations):
        np.random.shuffle(combined_data)
        X_permuted = combined_data[:n_x, :]
        
        Y_permuted = combined_data[n_x: , :] 
        permuted_mmd = mmd_rbf(X_permuted, Y_permuted)
        print(f"Trial {i + 1}: Permutated mmd = {permuted_mmd}")
        if permuted_mmd >= observed_mmd:
            count_extreme += 1
    print()
    print("Completed permutations")
    print(f"Number of extremes = {count_extreme}")
    p_value = count_extreme / n_permutations
    return p_value

# Example usage
if __name__ == "__main__":
    X_df1 = pd.read_csv("./data/X_train.csv")
    X_df2 = pd.read_csv("./data/X_test_2.csv")

    X = np.array(X_df1)
    Y = np.array(X_df2)

    # X = np.random.rand(100, 20)
    # Y = np.random.rand(100, 20) + 0.003  # Shifted distribution

    p = mmd_permutation_test(X, Y)
    print(f"Resulting p value is {p}")
