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
        # shuffle samples from each distribution
        np.random.shuffle(combined_data)
        X_permuted = combined_data[:n_x, :]
        
        Y_permuted = combined_data[n_x: , :] 
        # get the mmd value using Gaussian kernel
        permuted_mmd = mmd_rbf(X_permuted, Y_permuted)
        print(f"Trial {i + 1}: Permutated mmd = {permuted_mmd}")
        # compare to the original mmd value 
        if permuted_mmd >= observed_mmd:
            count_extreme += 1
    print()
    print("Completed permutations")
    print(f"Number of extremes = {count_extreme}")
    
    # p-value is calculated as the proportion of mmd values calculated that 
    # are greater than or equal to the observed mmd value to the total number of 
    # mmd values calculated
    p_value = count_extreme / n_permutations
    return p_value

if __name__ == "__main__":
    X_df1 = pd.read_csv("./data/X_train.csv")
    X_df2 = pd.read_csv("./data/X_test_2.csv")

    X = np.array(X_df1)
    Y = np.array(X_df2)

    p = mmd_permutation_test(X, Y)
    print(f"Resulting p value is {p}")
