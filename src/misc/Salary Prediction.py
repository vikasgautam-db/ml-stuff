# Databricks notebook source
df = spark.sql("select * from vikas_demo.default.salary_data")

df.printSchema()
col_list = ["age","gender","education_level", "job_title", "experience", "salary"]
renamed_df = df.toDF(*col_list)

renamed_df = renamed_df.dropna()

display(renamed_df)

# COMMAND ----------

def gender_to_bool(str):
  if(str == "Male"):
    return 0
  elif(str == "Female"):
    return 1
  elif(str == "Other"):
    return 2

# COMMAND ----------

def fix_education_level(str):
  if(str == "phD"):
    return "PhD"
  elif(str == "Bachelor's"):
    return "Bachelor's Degree"
  elif(str == "Master's"):
    return "Master's Degree"
  else:
    return str
  

# COMMAND ----------

display(renamed_df.select("education_level").distinct())

# COMMAND ----------

from pyspark.sql.functions import col, udf
from pyspark.sql.types import StringType

education_level = udf(lambda z: fix_education_level(z),StringType())

# COMMAND ----------

education_level_fix = renamed_df.withColumn("education_level_fixed", education_level(col("education_level")))

# COMMAND ----------

display(education_level_fix)

# COMMAND ----------

display(education_level_fix.summary())

# COMMAND ----------

education_level_fix.printSchema()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer
from pyspark.ml import Pipeline

indexer = StringIndexer(inputCol="job_title", outputCol="job_title_indexed")
indexer1 = StringIndexer(inputCol="education_level_fixed", outputCol="education_level_indexed")
indexer2 = StringIndexer(inputCol="gender", outputCol="gender_indexed")
pipeline = Pipeline(stages=[indexer, indexer1, indexer2])

# Fit and transform data
indexed_df = pipeline.fit(education_level_fix).transform(education_level_fix)

# Display encoded DataFrame
display(indexed_df)

# COMMAND ----------

final_df = indexed_df.select("age", "gender_indexed", "education_level_indexed", "job_title_indexed", "experience", "salary")

# COMMAND ----------

display(final_df)

# COMMAND ----------

train_df, test_df = final_df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
from pyspark.ml.feature import VectorAssembler

vec_assembler = VectorAssembler(inputCols=["age", "gender_indexed", "education_level_indexed", "job_title_indexed", "experience"], outputCol="features")

vec_train_df = vec_assembler.transform(train_df)
lr = LinearRegression(featuresCol="features", labelCol="salary")


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

display(pred_df)

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="salary", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
print(f"RMSE is {rmse}")

# COMMAND ----------

for col, coef in zip(vec_assembler.getInputCols(), lr_model.coefficients):
    print(col, coef)
  
print(f"intercept: {lr_model.intercept}")
