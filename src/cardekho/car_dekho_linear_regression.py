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

from pyspark.ml.feature import StringIndexer

indexer = StringIndexer(inputCol="fuel", outputCol="fuel_indexed")
fuel_indexed = indexer.fit(df).transform(df)
display(fuel_indexed)

# COMMAND ----------

indexer = StringIndexer(inputCol="seller_type", outputCol="seller_type_indexed")
st_indexed = indexer.fit(fuel_indexed).transform(fuel_indexed)
display(st_indexed)

# COMMAND ----------

indexer = StringIndexer(inputCol="transmission", outputCol="transmission_indexed")
trans_indexed = indexer.fit(st_indexed).transform(st_indexed)
display(trans_indexed)

# COMMAND ----------

from pyspark.sql.functions import col

df_final = trans_indexed.withColumn("years_in_service", (2023 - col("year")))

display(df_final)

# COMMAND ----------

df_final.printSchema()

# COMMAND ----------

train_df, test_df = df_final.randomSplit([.8, .2], seed=42)

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["km_driven", "years_in_service", "seller_type_indexed", "fuel_indexed", "transmission_indexed"], outputCol="features")

vec_train_df = vec_assembler.transform(train_df)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
lr = LinearRegression(featuresCol="features", labelCol="selling_price")
lr_model = lr.fit(vec_train_df)

# COMMAND ----------

m = lr_model.coefficients[0]
b = lr_model.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

vec_test_df = vec_assembler.transform(test_df)

pred_df = lr_model.transform(vec_test_df)

pred_df.select("name","selling_price", "prediction", "fuel_indexed", "years_in_service", "seller_type_indexed").show()

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="selling_price", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"RSquared is {r2}")

# COMMAND ----------


