import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

y = pd.read_csv("./data/y_test_2_reduced.csv")

class_counts = y['label'].value_counts().sort_index()
print("Class counts:\n", class_counts)

class_percentages = y['label'].value_counts(normalize=True).sort_index() * 100
print("Class percentages:\n", class_percentages)

plt.figure(figsize=(12, 6))
plt.bar(class_counts.index, class_counts.values)
plt.xlabel("Class Label")
plt.ylabel("Number of Samples")
plt.title("Class Distribution")
plt.xticks(class_counts.index) 
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()