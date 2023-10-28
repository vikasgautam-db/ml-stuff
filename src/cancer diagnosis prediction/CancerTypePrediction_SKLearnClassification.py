# Databricks notebook source
from sklearn.neighbors import KNeighborsClassifier
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import matplotlib.pyplot as plt

# COMMAND ----------

# import warnings filter
from warnings import simplefilter
# ignore all future warnings
simplefilter(action='ignore', category=FutureWarning)

# COMMAND ----------

df = pd.read_csv('/Workspace/Repos/vikas.gautam@databricks.com/ml-stuff/data/KNNAlgorithmDataset.csv')
df.columns

# COMMAND ----------

X = df[['radius_mean', 'texture_mean', 'perimeter_mean','area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave points_mean', 
        'symmetry_mean', 'fractal_dimension_mean','radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se','compactness_se', 
        'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 
        'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst','symmetry_worst', 'fractal_dimension_worst']].values
        
y = df['diagnosis'].values
print(X.shape, y.shape)

# COMMAND ----------

X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.3, random_state=21, stratify=y)

# COMMAND ----------

# testing accuracies

train_accuracies = {}
test_accuracies = {}
neighbours = np.arange(1, 26)
for neighbour in neighbours:
    knn = KNeighborsClassifier(n_neighbors = neighbour)
    knn.fit(X_train, y_train)
    train_accuracies[neighbour] = knn.score(X_train, y_train)
    test_accuracies[neighbour]= knn.score(X_test, y_test)

# COMMAND ----------

plt.figure(figsize=(10, 6))
plt.title("KNN: Varying Number of Neighbors")
plt.plot(neighbours, train_accuracies.values(), label="Training Accuracy")
plt.plot(neighbours, test_accuracies.values(), label="Testing Accuracy")
plt.legend()
plt.xlabel("Number of Neighbors")
plt.ylabel("Accuracy")

plt.show()

# COMMAND ----------


