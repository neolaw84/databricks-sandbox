# Databricks notebook source
# MAGIC %pip install "mlflow-skinny[databricks]>=2.4.1"
# MAGIC dbutils.library.restartPython()

# COMMAND ----------

import mlflow
catalog = "main"
schema = "default"
model_name = "my_model"
mlflow.set_registry_uri("databricks-uc")
mlflow.register_model("runs:/c28de582a91a48b4a9304323de43e0c3/nameless", f"{catalog}.{schema}.{model_name}")

# COMMAND ----------

!/databricks/python3/bin/databricks configure

# COMMAND ----------

!databricks unity-catalog metastores list

# COMMAND ----------


