# Databricks notebook source
df = spark.sql("select * from vikas_demo.car_dekho.car_details_from_car_dekho")
display(df)


df.printSchema()

# COMMAND ----------

display(df.select("year", "selling_price", "km_driven", "fuel", "transmission"))

# COMMAND ----------

from pyspark.sql.functions import col
df = df.filter(col("km_driven") < 200000).filter(col("selling_price") < 1000000)

display(df.select("selling_price", "km_driven").summary())

# COMMAND ----------

from pyspark.sql.functions import col

df_final = df.withColumn("years_in_service", (2023 - col("year")))

display(df_final)

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

categorical_cols = [field for (field, dataType) in df_final.dtypes if dataType == "string" and field != "name"]

index_output_cols = [x + "_index" for x in categorical_cols]

index_output_cols

# COMMAND ----------

indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols,handleInvalid="skip")
fuel_indexed = indexer.fit(df_final).transform(df)
display(fuel_indexed)

# COMMAND ----------

df_final.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

numeric_cols = [field for (field, dataType) in df_final.dtypes if ((dataType != "string") and (field != "selling_price"))]

numeric_cols

# COMMAND ----------

assembler_inputs = index_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(labelCol="selling_price")

# COMMAND ----------

train_df, test_df = df_final.randomSplit([.8, .2], seed=42)

# COMMAND ----------

from pyspark.ml import Pipeline

# Combine stages into pipeline
stages = [indexer, vec_assembler, dt]
pipeline = Pipeline(stages=stages)

# Uncomment to perform fit
pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

# COMMAND ----------

display(pred_df.select("features", "selling_price", "prediction").orderBy("selling_price", ascending=False))

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="selling_price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------


