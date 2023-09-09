# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 03 Model Registration
# MAGIC
# MAGIC In this notebook, we will **register** a developed model in the unity catalog. This will enable us to **serve** the model as an API end-point on `databricks` platform.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Preparations

# COMMAND ----------

# MAGIC %md 
# MAGIC **Install `mlflow-skinny`**
# MAGIC
# MAGIC To register a model, we need to install `mlflow-skinny` package.
# MAGIC
# MAGIC > Notice how we **restart** python runtime WITHOUT restarting the whole cluster.

# COMMAND ----------

# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **`import` necessary libraries**

# COMMAND ----------

import mlflow
import yaml

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Get `CONSTANTS`**
# MAGIC
# MAGIC We hard-code the **CONSTANTS** here.
# MAGIC
# MAGIC > We will see how we can get secrets and other environment variables when we do inferencing.

# COMMAND ----------

SELECTED_MODEL = "gbm_fast_large"
RUN_IDS_FILE_PATH = "../configurations/run_ids.yaml"

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Get `run_id`**
# MAGIC
# MAGIC Read the yaml file to obtain the run-id

# COMMAND ----------

with open(RUN_IDS_FILE_PATH, "rt") as f:
    run_ids = yaml.load(f, Loader=yaml.SafeLoader)
run_id = run_ids["run_id"][SELECTED_MODEL]

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Register the model
# MAGIC
# MAGIC Finally, we register the model, produced by our experiment (identified by `run_id`) and the name of the run (identified by `SELECTED_MODEL`). 

# COMMAND ----------

catalog = "ml"
schema = "default"
model_name = "gbm_fast_large"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model(f"runs:/{run_id}/{SELECTED_MODEL}", f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC > Databricks recommend to use three level namespace (i.e. `catalog`, `schema` and `model_name`) to be used to identify which stage (dev/staging/prod) the model is in. We will update after we have an update on the naming/notation convention.

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC ## Summary
# MAGIC
# MAGIC In this notebook, we see how we can register a model.
# MAGIC
# MAGIC > In practice, this part can be easily a part of the `model_development` notebook, where data scientist identify the best model/parameters and register right away (when `run_id` and `run_name`/`SELECTED_MODEL` information are readily available).

# COMMAND ----------


