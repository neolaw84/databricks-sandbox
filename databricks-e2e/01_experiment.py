# Databricks notebook source
# MAGIC %md
# MAGIC # Experimentation Using Pure *python*
# MAGIC
# MAGIC In this notebook, we will see how we build a model without using any facility from databricks.
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

def run_with_python(args, model):
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
                # notice this print statement
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                    epoch, batch_idx * len(data), len(train_loader.dataset),
                    100. * batch_idx / len(train_loader), loss.data.item()))
        model.eval()
        test_loss, correct = test_epoch(model, test_loader, args)
        # notice this print statement
        
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, len(test_loader.dataset),
            100. * correct / len(test_loader.dataset)))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 1 : default model, slower learning rate

# COMMAND ----------

run_with_python(args2, model_d)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 2 : default model, faster learning rate

# COMMAND ----------

run_with_python(args1, model_dd)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 3 : smaller model, slower learning rate

# COMMAND ----------

run_with_python(args2, model_s)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Experiment 4 : smaller model, faster learning rate

# COMMAND ----------

run_with_python(args1, model_ss)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Conclusion
# MAGIC
# MAGIC We learnt that tracking experiment manually is error prone. It is also possible to have undetected errors and mistakes. 

# COMMAND ----------


