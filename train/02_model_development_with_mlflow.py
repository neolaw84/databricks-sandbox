# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 02 Model Development with *`mlflow`*
# MAGIC
# MAGIC In this notebook, we will see how a typical model development works using `mlflow`. `mlflow` is a facility provided by default in a databricks environment. 

# COMMAND ----------

# MAGIC %md
# MAGIC ## Problem
# MAGIC We will use `california_housing` dataset to train models that can predict whether a given house is in top 25% most expensive houses.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparations

# COMMAND ----------

# change 1 : we need to import mlflow
import mlflow

import yaml 

import pandas as pd

from sklearn import datasets
from sklearn import metrics
from sklearn import model_selection
from sklearn import linear_model, ensemble

# COMMAND ----------

df_X, ds_y = datasets.fetch_california_housing(return_X_y=True, as_frame=True)

X_tr, X_ts, y_tr, y_ts = model_selection.train_test_split(df_X, ds_y, test_size=0.2, random_state=42)
percentile_75 = y_tr.describe()["75%"]

y_tr_label = y_tr >= percentile_75
y_ts_label = y_ts >= percentile_75

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

def train_and_test_with_mlflow(model, run_name, tr_X = X_tr, tr_label = y_tr_label, ts_X = X_ts, ts_label = y_ts_label):
    # change 2 : use mlflow.start_run context manager
    with mlflow.start_run(run_name=run_name):
        model.fit(tr_X, tr_label)
        y_pred = model.predict(ts_X)
        # change 3 : log the metrics (instead of or in addition to print)
        mlflow.log_metric(key="precision", value=metrics.precision_score(ts_label, y_pred))
        # print (f"Precision : {metrics.precision_score(ts_label, y_pred)}")
        mlflow.log_metric(key="recall", value=metrics.recall_score(ts_label, y_pred))
        # print (f"Recall : {metrics.recall_score(ts_label, y_pred)}")
        mlflow.log_metric(key="f1", value=metrics.f1_score(ts_label, y_pred))
        # print (f"f1 score : {metrics.f1_score(ts_label, y_pred)}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Training and Testing

# COMMAND ----------

train_and_test_with_mlflow(model_log_reg, "log_reg")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **More than just metrics**
# MAGIC
# MAGIC Actually, `mlflow` can help us track parameters, datasets, and model artifacts in addition to just metrics.

# COMMAND ----------

conda_env = mlflow.sklearn.get_default_conda_env(include_cloudpickle=True)
with open("../configurations/conda_env.yaml", "wt") as f:
    yaml.dump(conda_env, f)

# COMMAND ----------

def train_and_test_with_mlflow(model, run_name, params, tr_X = X_tr, tr_label = y_tr_label, ts_X = X_ts, ts_label = y_ts_label):
    # change 2 : use mlflow.start_run context manager
    with mlflow.start_run(run_name=run_name) as run:
        # change 3.2 : log the parameters
        mlflow.log_params(params)

        # change 3.3 log the training dataset
        mlflow.log_input(mlflow.data.from_pandas(tr_X), context="training-X")
        mlflow.log_input(mlflow.data.from_numpy(tr_label.to_numpy()), context="training-y")

        model.fit(tr_X, tr_label)
        y_pred = model.predict(ts_X)
        # change 3.1 : log the metrics
        mlflow.log_metrics({
            "precision" : metrics.precision_score(ts_label, y_pred),
            "recall" : metrics.recall_score(ts_label, y_pred),
            "f1" : metrics.f1_score(ts_label, y_pred)
        })
        
        # change 3.4 : log the model itself
        mlflow.sklearn.log_model(
            model, run_name,
            conda_env="../configurations/conda_env.yaml",
            signature=mlflow.models.infer_signature(tr_X, y_pred)
        )

        # change 4 : record the run_id
        run_id = run.info.run_id
    return run_id

# COMMAND ----------

run_id_log_reg = train_and_test_with_mlflow(model_log_reg, run_name="log_reg_2", params=param_log_reg)

# COMMAND ----------

run_id_svm = train_and_test_with_mlflow(model_svm, run_name="svm", params=param_svm)

# COMMAND ----------

run_id_gbm_fast_large = train_and_test_with_mlflow(model_fast_large, run_name="gbm_fast_large", params=param_fast_large)

# COMMAND ----------

run_id_gbm_slow_small = train_and_test_with_mlflow(model_slow_small, run_name="gbm_slow_small", params=param_slow_small)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Even how metrics progresses**
# MAGIC
# MAGIC In the example below, we will illustrate how `mlflow` can be used to track progression of metrics.

# COMMAND ----------

from sklearn import neural_network

net = neural_network.MLPClassifier(
    hidden_layer_sizes=(20, 20), 
    max_iter=25,
    warm_start=True
)

warm_start=False

with mlflow.start_run(run_name="neural_network"):
    for i in range(0, 10):
        net.fit(X_tr, y_tr_label)
        y_pred = net.predict(X_ts)
        mlflow.log_metrics({
            "precision" : metrics.precision_score(y_ts_label, y_pred),
            "recall" : metrics.recall_score(y_ts_label, y_pred),
            "f1" : metrics.f1_score(y_ts_label, y_pred)
        }, step=net.n_iter_)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Log the `run_id`
# MAGIC
# MAGIC Now, let us log the run_ids for cases when we want to register and serve one of these models.

# COMMAND ----------

with open("../configurations/run_ids.yaml", "wt") as f:
    yaml.dump({
        "run_id" : {
            "gbm_fast_large": run_id_gbm_fast_large,
            "gbm_slow_small": run_id_gbm_slow_small,
            "log_reg": run_id_log_reg,
            "svm": run_id_svm
        }
    }, f)

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC > Do not worry if you miss this step, run_ids are also accessible from databricks UI.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC We learnt that tracking the models, (training) parameters and (test) performance as well as datasets using **`mlflow`** is easy and straightforward. It reduces errors and is less cumbersome. Finally, it facilitates recording of experiments with minimum changes to workflows. 

# COMMAND ----------


