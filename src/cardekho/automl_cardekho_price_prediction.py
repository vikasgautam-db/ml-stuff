# Databricks notebook source
df = spark.sql("select * from vikas_demo.car_dekho.car_details_from_car_dekho")

# COMMAND ----------

from pyspark.sql.functions import col
df = df.filter(col("km_driven") < 200000).filter(col("selling_price") < 1000000)

# COMMAND ----------

from pyspark.sql.functions import col

df_final = df.withColumn("years_in_service", (2023 - col("year"))).dropna().drop("year")

display(df_final)

# COMMAND ----------

train_df, test_df = df_final.randomSplit([.8, .2], seed=42)

# COMMAND ----------

from databricks import automl

summary = automl.regress(train_df, target_col="selling_price", primary_metric="r2", timeout_minutes=10, max_trials=20)

# COMMAND ----------

print(summary.best_trial)

# COMMAND ----------

# Load the best trial as an MLflow Model
import mlflow

model_uri = f"runs:/{summary.best_trial.mlflow_run_id}/model"

predict = mlflow.pyfunc.spark_udf(spark, model_uri)
pred_df = test_df.withColumn("prediction", predict(*test_df.drop("selling_price").columns)[0])
display(pred_df)

# COMMAND ----------

display(pred_df.select("selling_price", "prediction"))

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="selling_price", metricName="rmse")
rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE on test dataset: {rmse:.3f}")
print(f"RSquared on test dataset: {r2:.3f}")

# COMMAND ----------


