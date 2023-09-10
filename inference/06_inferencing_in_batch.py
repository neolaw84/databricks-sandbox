# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 06 Inferencing in Batch
# MAGIC
# MAGIC In this notebook, we will try to develop a notebook that runs inference on a batch of new data.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Preparations

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **`import` necessary libraries**

# COMMAND ----------

import mlflow

import pandas as pd

from datetime import datetime as dt

from sklearn import datasets
from sklearn import model_selection

from pyspark.sql.functions import struct, col

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Retrieve our model**
# MAGIC
# MAGIC Then, we will retrieve our model.

# COMMAND ----------

mlflow.set_registry_uri("databricks-uc")
model_name = "ml.default.gbm_fast_large"
model_version = "1"
model = mlflow.pyfunc.spark_udf(spark, model_uri=f"models:/{model_name}/{model_version}", result_type='bool')

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Get new data**
# MAGIC
# MAGIC To simulate getting new data, we will **sample** the test split of the data. 
# MAGIC
# MAGIC > In real-life, it can be retrieved from a `feature_store`.

# COMMAND ----------

df_X, ds_y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)

X_tr, X_ts, y_tr, y_ts = model_selection.train_test_split(df_X, ds_y, test_size=0.2, random_state=42)
percentile_75 = y_tr.describe()["75%"]

y_tr_label = y_tr >= percentile_75
y_ts_label = y_ts >= percentile_75

random_seed = dt.now().second + 42
X_sample = X_ts.sample(n=5, random_state = random_seed)
display(X_sample)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Convert data to `spark`**

# COMMAND ----------

spark.conf.set("spark.sql.execution.arrow.pyspark.enabled", "true")

# Create a Spark DataFrame from a pandas DataFrame using Arrow
df = spark.createDataFrame(X_sample)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Invoke the model

# COMMAND ----------

# Predict on a Spark DataFrame.
results = df.withColumn('predictions', model(struct(*map(col, df.columns))))

# COMMAND ----------

display(results)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learnt how new batch of data can be inferred using a model in our `unity_catalog`. Furthermore, using the `Workflows` tab in the left sidebar, we can even run it as a job.

# COMMAND ----------


