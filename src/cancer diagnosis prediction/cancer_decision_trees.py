# Databricks notebook source
# MAGIC %sql 
# MAGIC
# MAGIC select * from vikas_demo.ml.breast_cancer;

# COMMAND ----------

# MAGIC %sql 
# MAGIC
# MAGIC select diagnosis, `concave points_mean` as concave_points_mean, radius_mean from vikas_demo.ml.breast_cancer;

# COMMAND ----------

df = spark.sql("select * from vikas_demo.ml.breast_cancer")

# COMMAND ----------

train_df, test_df = df.randomSplit([.8, .2], seed=42)

# COMMAND ----------

display(train_df)

# COMMAND ----------

train_df.printSchema()

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler

# Filter for just numeric columns (and exclude price, our label)
numeric_cols = [field for (field, dataType) in train_df.dtypes if ((dataType == "double") & (field != "diagnosis") & (field != "id"))]
# Combine output of StringIndexer defined above and numeric columns
assembler_inputs = numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")

# COMMAND ----------

from pyspark.ml.regression import DecisionTreeRegressor

dt = DecisionTreeRegressor(labelCol="diagnosis")

# COMMAND ----------

from pyspark.ml import Pipeline

# Combine stages into pipeline
stages = [vec_assembler, dt]
pipeline = Pipeline(stages=stages)

# COMMAND ----------

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------


