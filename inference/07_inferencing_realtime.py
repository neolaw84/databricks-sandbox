# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 07 Inferencing Realtime
# MAGIC
# MAGIC In this notebook, we will see how we can invoke the model as a RESTful API.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Preparations

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **`import` necessary libraries**

# COMMAND ----------

import os
import requests
import numpy as np
import pandas as pd
import json

from datetime import datetime as dt

from sklearn import datasets, model_selection

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Get CONSTANTS values**

# COMMAND ----------

DATABRICKS_URL="https://dbc-188baba4-9c32.cloud.databricks.com/"
ACCESS_TOKEN = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().getOrElse(None)
assert ACCESS_TOKEN

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Prepare invokation functions**

# COMMAND ----------

def create_tf_serving_json(data):
    return {'inputs': {name: data[name].tolist() for name in data.keys()} if isinstance(data, dict) else data.tolist()}

def score_model(dataset, ACCESS_TOKEN, url=f'{DATABRICKS_URL}serving-endpoints/high-end-housing/invocations'):
    headers = {'Authorization': f'Bearer {ACCESS_TOKEN}', 
'Content-Type': 'application/json'}
    ds_dict = {'dataframe_split': dataset.to_dict(orient='split')} if isinstance(dataset, pd.DataFrame) else create_tf_serving_json(dataset)
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.request(method='POST', headers=headers, url=url, data=data_json)
    if response.status_code != 200:
        raise Exception(f'Request failed with status {response.status_code}, {response.text}')

    return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Get random data**

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
# MAGIC ## Score the model through `http` endpoint

# COMMAND ----------

score_model(X_sample, ACCESS_TOKEN)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we learn how `databricks` model serving endpoint can be invoked from `python`. Similarly, we can develop `curl`, `Java` or other programming language to invoke our models' endpoints.

# COMMAND ----------


