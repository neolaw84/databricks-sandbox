# Databricks notebook source
# MAGIC %md
# MAGIC
# MAGIC # 05 `databricks` Secrets
# MAGIC
# MAGIC In this notebook, we will try to create a `databricks` secret. 

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC To enable using secrets (such as credentials and passwords), we need to use databricks `secrets`.
# MAGIC
# MAGIC Here we will try to put a databricks ACCESS_TOKEN in databricks `secrets`.
# MAGIC
# MAGIC The following steps explain how we can achieve it.

# COMMAND ----------

# MAGIC %md 
# MAGIC
# MAGIC **Create `ACCESS_TOKEN`**
# MAGIC
# MAGIC Before the next step. We need to prepare an ACCESS_TOKEN.
# MAGIC
# MAGIC 1. First, click your username (usually e-mail address) at the top right corner. Then, go to "User Settings".
# MAGIC   > If username is not there, click the down-arrow symbol there.
# MAGIC
# MAGIC   <img src="https://516237376-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-MIIWE47MSPMOIxmLgUz%2F-MIJ44Lg012rRHWFBOBg%2F-MIJ4hhU4-otaAlEJ_ox%2F25.png?alt=media&token=ddaaf8d5-9c03-41d6-a478-08ab077c3f13" width="80%"/> 
# MAGIC
# MAGIC 2. Then, go to "Access Token" tab. Click generate new token.
# MAGIC   > **WARNING** treat it as a password!!!
# MAGIC
# MAGIC   <img src="https://516237376-files.gitbook.io/~/files/v0/b/gitbook-legacy-files/o/assets%2F-MIIWE47MSPMOIxmLgUz%2F-MIJ44Lg012rRHWFBOBg%2F-MIJ5axFsB85O9DGZ5Tx%2F26.png?alt=media&token=81e5a54b-34a3-4f81-a95d-4c1065b574e9" width="80%">
# MAGIC
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **Put `ACCESS_TOKEN` in secrets**
# MAGIC
# MAGIC Next, we need to access a databricks CLI. Easiest way to achieve it is by clicking the "Bottom panel" button at the bottom right.
# MAGIC
# MAGIC > If you have not configure databricks CLI before, use the following command:
# MAGIC > ```bash
# MAGIC > databricks configure
# MAGIC > # you will need to put in databriks url (starting with https:// to databricks.com inclusive)
# MAGIC > ```
# MAGIC
# MAGIC ```bash
# MAGIC databricks secrets create-scope --scope e2e 
# MAGIC  databricks secrets put --scope e2e --key <YOUR ACCESS TOKEN HERE>
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC
# MAGIC **get the `ACCESS_TOKEN` back** 
# MAGIC
# MAGIC Now, we will retrieve the `ACCESS_TOKEN` back from the secrets. 
# MAGIC
# MAGIC > You won't get secrets as plain text when you `print` them out. 

# COMMAND ----------

ACCESS_TOKEN = dbutils.secrets.get(scope="e2e", key="ACCESS_TOKEN")
assert ACCESS_TOKEN

# COMMAND ----------


