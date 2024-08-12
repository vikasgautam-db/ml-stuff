# Databricks notebook source
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# Generate some random data
np.random.seed(0)
X = np.random.rand(100, 1)
y = 2 + 3 * X + np.random.randn(100, 1)

# Create polynomial features
poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

# Create and train the model
model = LinearRegression()
model.fit(X_poly, y)

# Set the default experiment
mlflow.set_experiment("/Workspace/Users/vikas.gautam@databricks.com/vgd_polynomial_regression")

# Log the model and parameters with MLflow
with mlflow.start_run():
    mlflow.log_params({"degree": 2})
    mlflow.sklearn.log_model(model, "model")

# Load the model from MLflow
runs = mlflow.search_runs()
run_id = runs.loc[0, "run_id"]
loaded_model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")

# Predict using the loaded model
X_test = np.array([[0.5]])
X_test_poly = poly.transform(X_test)
y_pred = loaded_model.predict(X_test_poly)

print("Predicted value:", y_pred)

# COMMAND ----------


