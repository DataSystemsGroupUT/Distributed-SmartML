# Documentation

## HIGH LEVEL DESIGN
The library has the following main component:

![Output](https://raw.githubusercontent.com/DataSystemsGroupUT/Distributed-SmartML/master/Images/design.png)
<ul>

<li>
<b>Model Selector</b>:<br/>
Responsible for Receiving Dataset and return the best Machine learning model found within the specified time budget.
It exposes a function “getBestModel” that start the Algorithm selection & Hyper-parameters optimization process.
</li>
<li>
<b>Meta Data Manager</b>:<br/>
Responsible for extracting the Dataset metadata (the data set characteristics) and produce a metadata object represent the dataset characteristic
The created object similar to Knowledgebase instances
</li>
<li>
<b>KB Manager</b>:<br/>
Responsible for handling all Knowledgebase activates like (Load Knowledge base, update knowledge base, …)
But the most important role, is to determine the suitable algorithms based on the dataset meta data and the loaded Knowledge base
</li>
<li>
<b>Classifier Manager</b>:<br/>
Represent all Classifiers and their parameters.
It contains the distribution for each parameter, to be used in random search and hyperband
It contains hyper-parameters range to be used with Grid Search
</li>
<li>
<b>KB Model</b>:<br/>
The mode that has been built based on the knowledge base and use to predict the suitable classifier(s)
</li>
<li>
<b>Grid Search</b>:<br/>
Responsible for doing Grid Search algorithm to do hyper parameter optimization
</li>
<li>
<b>Random Search</b>:<br/>
Responsible for doing Random Search algorithm to do hyper parameter optimization
</li>
<li>
<b>Hyperband</b>:<br/>
Responsible for doing hyperband algorithm to do hyper parameter optimization
</li>
</ul>


## PROCESSING SEQUENCE

To get the best model for input dataset, the library executes the following sequence:
<ol>
<li>“Model Selector” receive the dataset and call “KB Manger” to determine best classifiers suitable to the dataset.</li>
<li>“KB Manager” Load KB Model and call “Meta data manger” to extract metadata</li>
<li>“KB Manager” receive Metadata object and use the loaded model to predict suitable classifier then return Classifier List</li>
<li>“Model Selector” loop on the classifiers list and call “Classifier manger” to get the hyperparameters for each classifier (and their values distribution or grid)</li>
<li>“Model Selector” call “Hyperband” or “Random Search” or “Grid Search” and send the Classifier and its hyper parameters (distribution or grid) </li>
</ol>
![Output](https://raw.githubusercontent.com/DataSystemsGroupUT/Distributed-SmartML/master/Images/Flow.png)
## Parameters

| Parameter| Description | Data Type | Default Value |
| ------ | ------ |------ |------ |
| **eta**| an input that controls the proportion of configurations discarded in each round of  SuccessiveHalving (in hyperband) | Integer | 5 |
|**Max Data Percentage**|the maximum amount of resource that can be allocated to a single configuration|Integer| 100 |
|**Parallelism**|the maximum amount of resource that can be allocated to a single configuration (models will only be run in parallel if there are enough resources available in the cluster. Otherwise, models will be queued in the Spark scheduler and have to wait for the current jobs to complete before being run.)|Integer|1|
|**Try N Classifier**|Maximum Number of Algorithms should be checked (out of the best algorithms based on the kB)|Integer|2|
|**Max Time**|Maximum Time allowed for hyper parameter optimization (per each Algorithm) to get the best hyperparameter values  (in Seconds)| Integer|1800|
|**HP Optimizer**|Hyper parameters optimizer (1: Random Search or 2: Hyperband)|Integer|2|
|**Convert To Vector Assembly**|If the input dataset features need to be converted to Vector or not|Boolean|false|



