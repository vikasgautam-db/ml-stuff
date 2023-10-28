# Databricks notebook source
# MAGIC %sql
# MAGIC
# MAGIC select * from vikas_demo.hotels.all_hotels;

# COMMAND ----------

from pyspark.sql.functions import *

# COMMAND ----------

df = spark.sql("select * from vikas_demo.hotels.all_hotels")
display(df)

# COMMAND ----------

from pyspark.sql.functions import col, regexp_replace

df_clean = df.withColumn("num_reviews", regexp_replace(col("Reviews"), "reviews", ""))
df_clean = df_clean.withColumn("num_reviews", regexp_replace(col("num_reviews"), ",", ""))

df_clean = df_clean.drop("Reviews")

# COMMAND ----------

display(df_clean)

# COMMAND ----------

import re

def extract_price(str):
  str = str.replace(",", "")
  if(str.find(" Current price ") != -1):
    result = re.search("Current price" + r".*?([\d]+)", str)
    return(result.group(1))
  else:
    result = re.findall(r'\d+', str)
    return(result[0])

  

# COMMAND ----------

extract_price("Original price 533 zł. Current price 1,435 zł.")

extract_price("Price 334 zł")

# COMMAND ----------

from pyspark.sql.functions import col, udf

get_per_night_price = udf(lambda z: extract_price(z),StringType())

# COMMAND ----------

display(df.select("Price", get_per_night_price(col("Price")).alias("price_per_night")))

# COMMAND ----------

df_clean = df_clean.withColumn("price_per_night", get_per_night_price(col("Price")))

df_clean = df_clean.drop("Price")

# COMMAND ----------

display(df_clean)

display(df_clean.select("Performances").distinct())

# COMMAND ----------

display(df_clean.select("Performances").groupBy("Performances").count())

# COMMAND ----------

#df_filtered = df_clean.filter(col('Performances') != "Wonderful 9.0" & col('Performances') != "Review score 6.2" & col('Performances') != "Exceptional 10")

df_filtered = df_clean.filter((col('Performances') != "Wonderful 9.0") & (col('Performances') != "Review score 6.2") & (col('Performances') != "Exceptional 10"))

display(df_filtered.groupBy("Performances").count())


# COMMAND ----------

display(df_filtered)

# COMMAND ----------

def breakfast_fix(str):
  if(str == None):
    return 0
  result = 0
  if(str.find("Breakfast") != -1):
    result = 1
  return result

def dinner_fix(str):
  if(str == None):
    return 0
  result = 0
  if(str.find("Dinner") != -1 & str.find("dinner") != -1):
    result = 1
  return result

# COMMAND ----------

from pyspark.sql.functions import col, udf

get_breakfast = udf(lambda z: breakfast_fix(z),StringType())

get_dinner = udf(lambda z: dinner_fix(z),StringType())

# COMMAND ----------

df_with_bfast_dinner_fix = df_filtered.withColumn("breakfast_included", get_breakfast(col("Breakfast"))).withColumn("dinner_included", get_dinner(col("Breakfast")))

df_with_bfast_dinner_fix = df_with_bfast_dinner_fix.drop("Breakfast")

# COMMAND ----------

display(df_with_bfast_dinner_fix)

# COMMAND ----------

def get_distance_from_center(str):
  if(" km from center" in str):
    result = re.findall(r'([\d.]+)\s', str)[0]
  elif(" m from center" in str):
    result = int(re.findall(r'([\d.]+)\s', str)[0])/1000
  else:
    result = 10
  
  return result

# COMMAND ----------

get_distance_from_center("190.2 km from center")


# COMMAND ----------

from pyspark.sql.functions import col, udf

get_distance = udf(lambda z: get_distance_from_center(z),StringType())

# COMMAND ----------

df_with_distance_fix = df_with_bfast_dinner_fix.withColumn("distance_from_center", get_distance(col("Distances")))

df_with_distance_fix = df_with_distance_fix.drop("Distances")
display(df_with_distance_fix.filter(col("distance_from_center") < 0.1))

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder
from pyspark.ml import Pipeline

# Create StringIndexer
indexer = StringIndexer(inputCol="Performances", outputCol="Performances_indexed")

# Create OneHotEncoder
encoder = OneHotEncoder(inputCol="Performances_indexed", outputCol="perf_encoded")

# Create pipeline
pipeline = Pipeline(stages=[indexer, encoder])

# Fit and transform data
encoded_df = pipeline.fit(df_with_distance_fix).transform(df_with_distance_fix)

# Display encoded DataFrame
display(encoded_df)

# COMMAND ----------

display(df_with_distance_fix)

# COMMAND ----------

df_with_distance_fix.printSchema()

# COMMAND ----------

# Cast column to double
df_double = df_with_distance_fix.withColumn("price_per_night", col("price_per_night").cast("double")).withColumn("distance_from_center", col("distance_from_center").cast("double")).withColumn("breakfast_included", col("breakfast_included").cast("double"))

# COMMAND ----------

display(df_double.select("Hotel name", "Region City", "price_per_night", "distance_from_center", "Marks").orderBy("price_per_night"))

# COMMAND ----------

df_final = df_double.filter(col("price_per_night") < 600).filter(col("distance_from_center") < 10)

# COMMAND ----------

display(df_final)

# COMMAND ----------

display(df_final.select("price_per_night", "distance_from_center", "Stars", "breakfast_included").summary())

# COMMAND ----------

train_df, test_df = df_final.randomSplit([.8, .2], seed=42)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["distance_from_center", "Stars", "breakfast_included"], outputCol="features")

vec_train_df = vec_assembler.transform(train_df)
lr = LinearRegression(featuresCol="features", labelCol="price_per_night")


# COMMAND ----------

lr_model = lr.fit(vec_train_df)

# COMMAND ----------

m = lr_model.coefficients[0]
b = lr_model.intercept

print(f"The formula for the linear regression line is y = {m:.2f}x + {b:.2f}")

# COMMAND ----------

vec_test_df = vec_assembler.transform(test_df)

pred_df = lr_model.transform(vec_test_df)



# COMMAND ----------

display(pred_df.select("Hotel name", "Region City", "distance_from_center", "price_per_night", "prediction"))

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="price_per_night", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")
