# Model Integrity Framework

Model Integrity Framework (MIF) is the first and mandatory governance process for operationalisation of any developed model. 

Suncorp team already has place(s) for MIF. For example, there is confluence page(s) for each operational models. 

This section in the repo, which is supposed to be in version control system (VCS), will contain code and summary monitoring info to support MIF -- not to replace MIF.

> Strictly speaking, version control system (VCS) such as bitbucket, cannot host secrets, password, credentials and all types of data. Therefore, this section is limited to code and summary information. 

Therefore, in this directory, we will have the followings: 

1. A copy of the notebook that shows/produces initial validation of the model. 

2. A copy of the notebook that shows/produces key metrics, which are monitored regularly including:
  * Model output 
    * Distribution of prediction
    * Distribution of confidence
    * Overall model accuracy etc.
  * Model input
    * Distribution of input features

> **WARNING** Extra care should be given not to include secrets, password, credentials and all types of data. 
  It is ok to include a link to the unity catalog artifact (csv file etc.) as unity catalog is going to be access controlled.