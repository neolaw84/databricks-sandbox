# Databricks notebook source
# MAGIC %md
# MAGIC # Experimentation Using *mlflow*
# MAGIC
# MAGIC In this notebook, we will see how we build the same model using *mlflow* from databricks.
# MAGIC
# MAGIC In this exercise, we will use `pytorch` and `keras` hand-written digits dataset. 
# MAGIC
# MAGIC It is a classification problem with 10 classes.

# COMMAND ----------

# this cell, when uncomment and run, it will restart the databricks notebook 
# without requiring to restart the cluster and/or detach/re-attach the notebook to cluster

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Usual `import` statements
# MAGIC
# MAGIC First, we have usual `import` statements to import libraries. 
# MAGIC
# MAGIC

# COMMAND ----------

import torch

from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# COMMAND ----------

# MAGIC %md
# MAGIC Secondly, `mlflow` itself.

# COMMAND ----------

import mlflow

# COMMAND ----------

from experiment_utils import get_data_loader, Params, Net, train_epoch, test_epoch

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparation for experiments
# MAGIC
# MAGIC Next, we will prepare certain constants such as `USE_CUDA` and `RANDOM_SEED`. 

# COMMAND ----------

USE_CUDA = torch.cuda.is_available()
RANDOM_SEED = 42

torch.manual_seed(RANDOM_SEED)

# COMMAND ----------

# MAGIC %md
# MAGIC We are going to use two learning rates in this experiments.

# COMMAND ----------

# first learning rate experiment
args1 = Params(batch_size=64, test_batch_size=1000, epochs=10, lr=0.01, momentum=0.5, seed=RANDOM_SEED, cuda=USE_CUDA, log_interval=200)

# slow learning rate experiment
args2 = Params(batch_size=64, test_batch_size=1000, epochs=20, lr=0.005, momentum=0.5, seed=RANDOM_SEED, cuda=USE_CUDA, log_interval=200)

# COMMAND ----------

# MAGIC %md
# MAGIC ... and two sets of models, a default and a smaller model

# COMMAND ----------

# default model
model_d = Net()
model_d.share_memory() # gradients are allocated lazily, so they are not shared here

# COMMAND ----------

# smaller model
model_s = Net(num_conv_channels=[5, 10], last_fc_input=40)
model_s.share_memory()

# COMMAND ----------

# default model
model_dd = Net()
model_dd.share_memory() # gradients are allocated lazily, so they are not shared here

# smaller model
model_ss = Net(num_conv_channels=[5, 10], last_fc_input=40)
model_ss.share_memory()

print ("")

# COMMAND ----------

def run_with_mlflow(args, model, run_name=None):
    # change 1 ... with mlflow.start_run
    with mlflow.start_run(run_name=run_name):
        # Run the training loop over the epochs (evaluate after each)
        if args.cuda:
            model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        train_loader = get_data_loader(shuffle=False, batch_size=args.batch_size)
        test_loader = get_data_loader(train=False, shuffle=True, download=False, batch_size=args.test_batch_size)

        step = 0
        for epoch in range(1, args.epochs + 1):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                step = step + 1
                loss = train_epoch(model, data, target, optimizer)
                if batch_idx % args.log_interval == 0:
                    # change 2 ... instead of print, we log
                    mlflow.log_metric(key="train_loss", value=loss.data.item(), step=step)
                    # print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    #     epoch, batch_idx * len(data), len(train_loader.dataset),
                    #     100. * batch_idx / len(train_loader), loss.data.item()))
            model.eval()
            test_loss, correct = test_epoch(model, test_loader, args)
            # change 2 ... instead of print, we log
            mlflow.log_metric(key="average_loss", value=test_loss, step=step)
            mlflow.log_metric(key="num_correct", value=correct, step=step)
            # mlflow.log_metric(key="total_tested", value=len(test_loader.dataset), step=step)
            mlflow.log_metric(key="accuracy", value= 100. * correct / len(test_loader.dataset), step=step)
            # print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            #     test_loss, correct, len(test_loader.dataset),
            #     100. * correct / len(test_loader.dataset)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 1 : default model, slower learning rate

# COMMAND ----------

run_with_mlflow(args2, model_d, run_name="default-model-slower-lr")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 2 : default model, faster learning rate

# COMMAND ----------

run_with_mlflow(args1, model_dd, run_name="default-model-faster-lr")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 3 : smaller model, slower learning rate

# COMMAND ----------

run_with_mlflow(args2, model_s, run_name="smaller-model-slower-lr")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 4 : smaller model, faster learning rate

# COMMAND ----------

run_with_mlflow(args1, model_ss, run_name="smaller-model-faster-lr")

# COMMAND ----------

# MAGIC %md
# MAGIC ## More than metrics ...
# MAGIC
# MAGIC In fact, we can have mlflow track and store model parameters and model artifact themselves. 

# COMMAND ----------

from mlflow.models.signature import infer_signature
from mlflow.utils.environment import _mlflow_conda_env
import cloudpickle

class ModelWrapper(mlflow.pyfunc.PythonModel):
    def __init__(self, model):
        self.model = model
        
    # notice this need to be one at a time
    def predict(self, context, model_input):
        mi = torch.as_tensor(model_input)
        proba = F.softmax(self.model.forward(mi))
        return torch.argmax(proba, dim=1).cpu().detach().numpy()
    
def run_with_mlflow_all(args, model, run_name=None, model_name="nameless"):
    # change 1 ... with mlflow.start_run
    with mlflow.start_run(run_name=run_name):
        # change 3 ... parameters
        mlflow.log_params(args._asdict())
        if args.cuda:
            model = model.cuda()

        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

        train_loader = get_data_loader(shuffle=False, batch_size=args.batch_size)
        test_loader = get_data_loader(train=False, shuffle=True, download=False, batch_size=args.test_batch_size)

        step = 0
        for epoch in range(1, 2): # args.epochs + 1):
            model.train()
            for batch_idx, (data, target) in enumerate(train_loader):
                step = step + 1
                loss = train_epoch(model, data, target, optimizer)
                if batch_idx % args.log_interval == 0:
                    # change 2 ... instead of print, we log
                    mlflow.log_metric(key="train_loss", value=loss.data.item(), step=step)
            model.eval()
            test_loss, correct = test_epoch(model, test_loader, args)
            # change 2 ... instead of print, we log
            mlflow.log_metrics({
                "average_loss" : test_loss, 
                "num_correct" : correct, 
                "accuracy" : 100. * correct / len (test_loader.dataset)
            }, step=step)
            
        # change 4 ... wrap the model and store
        wrappedModel = ModelWrapper(model)
        # Log the model with a signature that defines the schema of the model's inputs and outputs. 
        # When the model is deployed, this signature will be used to validate inputs.
        data_to_infer = data.cpu().detach().numpy()
        output_to_infer = wrappedModel.predict(None, data_to_infer)
        signature = infer_signature(data_to_infer, output_to_infer)

        # MLflow contains utilities to create a conda environment used to serve models.
        # The necessary dependencies are added to a conda.yaml file which is logged along with the model.
        conda_env =  _mlflow_conda_env(
            additional_conda_deps=None,
            additional_pip_deps=[
                "cloudpickle=={}".format(cloudpickle.__version__),
                "torch=={}".format("1.13.1")], 
            additional_conda_channels=None,
        )
        mlflow.pyfunc.log_model(model_name, python_model=wrappedModel, conda_env=conda_env, signature=signature)

# COMMAND ----------

model_last = Net()
model_last.share_memory()
run_with_mlflow_all(args1, model_last, run_name="last_run", model_name="nameless")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC We learnt that tracking experiment using mlflow is much easier. It is also possible to have metrics charted and it is less prone to errors and mistakes. 

# COMMAND ----------


