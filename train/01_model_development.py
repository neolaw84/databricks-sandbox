# Databricks notebook source
# MAGIC %md
# MAGIC # 01 Model Development Using *Pure `python`*
# MAGIC
# MAGIC In this notebook, we will see how a typical model development works using pure `python`. We will not use any databricks specific facility. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem
# MAGIC We will use `california_housing` dataset to train models that can predict whether a given house is in top 25% most expensive houses.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparations

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Import necessary libraries**
# MAGIC
# MAGIC First, we will import necessary libraries.

# COMMAND ----------

import pandas as pd

from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import linear_model, ensemble

# COMMAND ----------

# MAGIC %md
# MAGIC **Fetch data**
# MAGIC
# MAGIC Next, we will fetch necessary data. 

# COMMAND ----------

df_X, ds_y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC > Notice `databricks` can visualise and profile data in notebook without writing code.

# COMMAND ----------

display(df_X)

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC **Train/Test split and labelling**
# MAGIC
# MAGIC Now, we will do train/test split and label the data.

# COMMAND ----------

X_tr, X_ts, y_tr, y_ts = model_selection.train_test_split(df_X, ds_y, test_size=0.2, random_state=42)
percentile_75 = y_tr.describe()["75%"]

y_tr_label = y_tr >= percentile_75
y_ts_label = y_ts >= percentile_75

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Models and parameters**
# MAGIC
# MAGIC Now, we will prepare some models and parameters. 
# MAGIC
# MAGIC For simplicity, we will use `SGDClassifier` and `GradientBoostingClassifier` with 2 sets of parameters each. This will give us a 4 combinations of models and parameters. 

# COMMAND ----------

param_log_reg = {"loss": "log_loss", "penalty": "elasticnet"}
param_svm = {"loss": "hinge", "penalty": "elasticnet"}

model_log_reg = linear_model.SGDClassifier(**param_log_reg)
model_svm = linear_model.SGDClassifier(**param_svm)

param_fast_large = {"n_estimators": 100, "learning_rate": 0.15}
param_slow_small = {"n_estimators": 80, "learning_rate": 0.1}

model_fast_large = ensemble.GradientBoostingClassifier(**param_fast_large)
model_slow_small = ensemble.GradientBoostingClassifier(**param_slow_small)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Getting ready to train/test**
# MAGIC
# MAGIC Finally, we prepare a utility function to train and test.

# COMMAND ----------

def train_and_test(model, tr_X = X_tr, tr_label = y_tr_label, ts_X = X_ts, ts_label = y_ts_label):
    print (model)
    model.fit(tr_X, tr_label)
    y_pred = model.predict(ts_X)
    print (f"Precision : {metrics.precision_score(ts_label, y_pred)}")
    print (f"Recall : {metrics.recall_score(ts_label, y_pred)}")
    print (f"f1 score : {metrics.f1_score(ts_label, y_pred)}")
    print ("Confusion Matrix")
    print (metrics.confusion_matrix(ts_label, y_pred))

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Training and Testing

# COMMAND ----------

train_and_test(model_log_reg)

# COMMAND ----------

train_and_test(model_svm)

# COMMAND ----------

train_and_test(model_fast_large)

# COMMAND ----------

train_and_test(model_slow_small)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC We learnt that tracking the models, (training) parameters and (test) performance using **pure python** is cumbersome and error-prone. 

# COMMAND ----------


