import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

# Load the datasets
X = pd.read_csv('./data/X_train.csv')
y = pd.read_csv('./data/y_train.csv')

# Combine into one DataFrame for EDA
df = X.copy()
df['label'] = y.values.ravel()  # flatten in case it's a DataFrame

print("Feature shape:", X.shape)
print("Label shape:", y.shape)
df.head()

# Exploratory Data Analysis

# Plot graph of class counts
class_counts = df['label'].value_counts().sort_index()

plt.figure(figsize=(12, 5))
sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis')
plt.title("Class Distribution")
plt.xlabel("Class Label")
plt.ylabel("Count")
plt.xticks(rotation=90)
plt.tight_layout()
plt.show()

print("Class counts:\n", class_counts)

# Summary stats for a few features
print("Summary stats:\n", X.iloc[:, :5].describe())

# Check for missing values
missing_total = X.isnull().sum().sum()
print(f"\nTotal missing values in X: {missing_total}")


# PLot correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(X.iloc[:, :300].corr(), cmap='coolwarm', annot=False)
plt.title("Correlation Heatmap of 300 features")
plt.show()


# Plot PCA graph

pca = PCA(n_components=2, random_state=42)
X_pca = pca.fit_transform(X)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=df['label'], palette='tab20', legend=False)
plt.title("PCA Projection of Feature Space")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()


