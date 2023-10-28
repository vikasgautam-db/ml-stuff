# Databricks notebook source
df = spark.sql("select * from vikas_demo.default.salary_data")

df.printSchema()
col_list = ["age","gender","education_level", "job_title", "experience", "salary"]
renamed_df = df.toDF(*col_list)

renamed_df = renamed_df.dropna()

display(renamed_df)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ### convert numeric columns to doubles

# COMMAND ----------

renamed_df.printSchema()

# COMMAND ----------

long_cols = [field for (field, dataType) in renamed_df.dtypes if dataType == "bigint"]
long_cols

# COMMAND ----------

from pyspark.sql.types import DoubleType
from pyspark.sql.functions import col, cast

for col_name in long_cols:
    renamed_df = renamed_df.withColumn(col_name, col(col_name).cast('double'))

# COMMAND ----------

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

# MAGIC %md
# MAGIC
# MAGIC ### use String Indexer and One Hot Encoding to include Categorical Columns in prediction

# COMMAND ----------

from pyspark.ml.feature import StringIndexer, OneHotEncoder

categorical_cols = [field for (field, dataType) in education_level_fix.dtypes if dataType == "string"]
index_output_cols = [x + "Index" for x in categorical_cols]
ohe_output_cols = [x + "OHE" for x in categorical_cols]

# COMMAND ----------

string_indexer = StringIndexer(inputCols=categorical_cols, outputCols=index_output_cols, handleInvalid="skip")
ohe_encoder = OneHotEncoder(inputCols=index_output_cols, outputCols=ohe_output_cols)

# COMMAND ----------

from pyspark.ml import Pipeline
from pyspark.ml.feature import VectorAssembler

numeric_cols = [field for (field, dataType) in education_level_fix.dtypes if ((dataType == "double") & (field != "salary"))]
assembler_inputs = ohe_output_cols + numeric_cols
vec_assembler = VectorAssembler(inputCols=assembler_inputs, outputCol="features")



# COMMAND ----------

from pyspark.ml.regression import LinearRegression

lr = LinearRegression(labelCol="salary", featuresCol="features")

# COMMAND ----------

pipeline = Pipeline(stages=[string_indexer, ohe_encoder, vec_assembler, lr])

# COMMAND ----------

train_df, test_df = education_level_fix.randomSplit([.8, .2], seed=42)

# COMMAND ----------



# COMMAND ----------

pipeline_model = pipeline.fit(train_df)

# COMMAND ----------

pred_df = pipeline_model.transform(test_df)

display(pred_df.select("features", "salary", "prediction"))

# COMMAND ----------

display(pred_df.select("salary", "prediction"))

# COMMAND ----------

from pyspark.ml.evaluation import RegressionEvaluator

regression_evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="salary", metricName="rmse")

rmse = regression_evaluator.evaluate(pred_df)
r2 = regression_evaluator.setMetricName("r2").evaluate(pred_df)
print(f"RMSE is {rmse}")
print(f"R2 is {r2}")

# COMMAND ----------


