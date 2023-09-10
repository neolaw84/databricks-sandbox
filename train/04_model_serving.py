# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 04 Model Serving
# MAGIC
# MAGIC In this notebook, we will try to serve a registered model as an API endpoint on `databricks` platform. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Requirements**
# MAGIC
# MAGIC * Only available for Python-based MLflow models registered in the MLflow Model Registry or Unity Catalog. 
# MAGIC * You must declare all model dependencies in the conda environment or requirements file.
# MAGIC
# MAGIC > If you use custom libraries or libraries from a private mirror server with your model (i.e. Suncorp environment), it will be *more complicated* but essentially the same flow.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Preparations

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **`import` the libraries**

# COMMAND ----------

import os
import requests
import pandas as pd

import json

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **get the `ACCESS_TOKEN` back** 
# MAGIC
# MAGIC Now, we will retrieve the `ACCESS_TOKEN` using `dbutils`. 
# MAGIC
# MAGIC > You won't get secrets as plain text when you `print` them out. 

# COMMAND ----------

ACCESS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
assert ACCESS_TOKEN

# COMMAND ----------

DATABRICKS_URL="https://dbc-188baba4-9c32.cloud.databricks.com/"

# COMMAND ----------

url = f'{DATABRICKS_URL}api/2.0/serving-endpoints'
print (url)
headers = {'Authorization': f'Bearer {ACCESS_TOKEN}'}
data_json = {
    "name": "high-end-housing",
    "config":{
        "served_models": [{
            "model_name": "ml.default.gbm_fast_large",
            "model_version": "1",
            "workload_size": "Small",
            "scale_to_zero_enabled": True
            }]
        }
    }
print (data_json)
response = requests.request(method='POST', headers=headers, url=url, json=data_json)
if response.status_code != 200:
    print(f'Request failed with status {response.status_code}, {response.text}')
else:
    print("all good.")

# COMMAND ----------


