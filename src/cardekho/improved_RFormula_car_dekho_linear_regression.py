# Databricks notebook source
df = spark.sql("select * from vikas_demo.car_dekho.car_details_from_car_dekho")

# COMMAND ----------

df.printSchema()

# COMMAND ----------

display(df.select("year", "selling_price", "km_driven", "fuel", "transmission"))

# COMMAND ----------

display(df.select("selling_price", "km_driven"))

# COMMAND ----------

from pyspark.sql.functions import col
df = df.filter(col("km_driven") < 200000).filter(col("selling_price") < 1000000)

# COMMAND ----------

display(df.select("selling_price", "km_driven").summary())

# COMMAND ----------

from pyspark.sql.functions import col

df_final = df.withColumn("years_in_service", (2023 - col("year"))).drop("year").dropna()

display(df_final)

# COMMAND ----------

df_final.printSchema()

# COMMAND ----------

train_df, test_df = df_final.randomSplit([.8, .2], seed=42)

# COMMAND ----------


from pyspark.ml import Pipeline
from pyspark.ml.feature import RFormula
from pyspark.ml.regression import LinearRegression
from pyspark.ml.evaluation import RegressionEvaluator

# COMMAND ----------

r_formula = RFormula(formula='selling_price ~ .', handleInvalid='skip', featuresCol="features")
lr = LinearRegression(labelCol="selling_price")

# COMMAND ----------

pipeline = Pipeline(stages=[r_formula, lr])
pipeline_model = pipeline.fit(train_df)
pred_df = pipeline_model.transform(test_df)

# COMMAND ----------

display(pred_df.select("selling_price", "prediction"))

# COMMAND ----------

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="selling_price")

rmse = regression_evaluator.setMetricName("rmse").evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------


