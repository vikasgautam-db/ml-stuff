# Databricks notebook source
df = spark.sql("select * from vikas_demo.car_dekho.car_details_from_car_dekho")

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.select("year", "selling_price", "km_driven", "fuel", "transmission"))

# COMMAND ----------

display(df.select("selling_price", "km_driven"))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### discarding extremes
# MAGIC

# COMMAND ----------

from pyspark.sql.functions import col
df = df.filter(col("km_driven") < 200000).filter(col("selling_price") < 1000000)

# COMMAND ----------

display(df.select("selling_price", "km_driven").summary())

# COMMAND ----------

from pyspark.sql.functions import col

df_final = df.withColumn("years_in_service", (2023 - col("year"))).dropna().drop("year")

display(df_final)

# COMMAND ----------

df_final.printSchema()

# COMMAND ----------

train_df, test_df = df_final.randomSplit([.8, .2], seed=42)

# COMMAND ----------

import mlflow
import mlflow.spark
from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import RFormula
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import RegressionEvaluator
from mlflow.models.signature import infer_signature

with mlflow.start_run(run_name="vg_car_dekho_lr") as run:
  mlflow.autolog()
  r_formula = RFormula(formula='selling_price ~ .', handleInvalid='skip', featuresCol="features")
  lr = LinearRegression(labelCol="selling_price")
  pipeline = Pipeline(stages=[r_formula, lr])
  pipeline_model = pipeline.fit(train_df)
  pred_df = pipeline_model.transform(test_df)

  # Log parameters
  #mlflow.log_param("label", "selling_price")
  #mlflow.log_param("features", "all_features")
   #(log_input_examples=True, log_model_signatures=True, log_models=True)

  # Log model
  #mlflow.spark.log_model(pipeline_model, "model", input_example=train_df.limit(5).toPandas())

  # Create predictions and metrics
  #pred_df = pipeline_model.transform(test_df)
  #signature = infer_signature(test_df)
  #regression_evaluator = RegressionEvaluator(labelCol="selling_price", predictionCol="prediction")
  #rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
  #r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)

  # Log both metrics
  #mlflow.log_metric("rmse", rmse)
  #mlflow.log_metric("r2", r2)


# COMMAND ----------

from mlflow.tracking import MlflowClient

client = MlflowClient()

# COMMAND ----------

experiment_id = run.info.experiment_id
runs_df = mlflow.search_runs(experiment_id)

display(runs_df)

# COMMAND ----------

model_path = f"runs:/{run.info.run_id}/model"
loaded_model = mlflow.spark.load_model(model_path)

display(loaded_model.transform(test_df).select("selling_price", "prediction"))

# COMMAND ----------

run_id = run.info.run_id
model_uri = f"runs:/{run_id}/model"

model_details = mlflow.register_model(model_uri=model_uri, name="vg_car_dekho_lr")

# COMMAND ----------

model_details

# COMMAND ----------

import mlflow.pyfunc

model_version_uri = f"models:/vg_car_dekho_lr/6"

print(f"Loading registered model version from URI: '{model_version_uri}'")
model_version_3 = mlflow.pyfunc.load_model(model_version_uri)

# COMMAND ----------

type(model_version_3)

# COMMAND ----------

import pandas as pd
display(spark.createDataFrame(pd.DataFrame(model_version_3.predict(test_df.toPandas()))))
