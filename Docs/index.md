# Welcome to D-Smart ML


## Introduction
D-SmartML is a  Scala-based distributed AutoML framework built on top of spark.ml (the  popular distributed big data processing framework
on computing clusters). 
<br/>
D-SmartML is equipped with a meta learning mechanism for automated algorithm selection and supports three different automated hyperparameter
tuning techniques: grid search, random search and hyperband optimization. 
<br/>
As a distributed framework, D-SmartML enables harnessing the power of computing clusters that have multiple nodes to tackle 
the computational complexity of the AutoML challenges on massive datasets.
<br/>

## Main Idea
The D-Smart ML library main objective is to do automatic machine learning in a distributed manner, 
by receiving a dataset as input and return the best Machine Learning model as output with minimum resources (ex: Time) 
without the need of selecting the algorithm or its hyperparameters manually.<br>
The library built on Apache Spark as a distributed processing platform, 
so the process of getting the best machine learning model can be easily distributed (like any spark application).<br><br>
The library main idea can be summarized in the below figure:

![Main Idea](https://raw.githubusercontent.com/DataSystemsGroupUT/Distributed-SmartML/master/Images/MainIdea.png)

The generation of the best machine learning model can be divided into two steps:
<ul>
<li>Selecting the best Algorithm</li>
<li>Selecting the best hyperparameters for the Selected Algorithm</li>
</ul>
The library contains a Knowledge base to support Algorithm selection, 
the knowledge based created based on 80 different datasets, it contains the meta data for each dataset and its performance against each classifier.<br>
The library uses Spark ML out of the box classifiers and also contains implementation for two new Classifiers 
(Linear Discernment Analysis and Quadratic Discernment Analysis).<br>
The library accepts multiple parameters to determine the desired behavior and most of the parameters have default value.<br>

![Main Idea](https://raw.githubusercontent.com/DataSystemsGroupUT/Distributed-SmartML/master/Images/Steps.png)

The process of determining best model consists of three steps as shown in the figure.
<ul>
<li>Metadata extraction</li>
<li>Algorithm Selection based on internal Knowledge base</li>
<li>Hyperparameter optimization using (Hyperband algorithm – Random Search – Grid Search)</li>
</ul>

## Our Knowledgebase
For each dataset, a set of meta-data extracted that represent statistics and description for this dataset, the extracted meta data can be grouped as following:<br/>
<ul>
<li>For All the Dataset:
	<ul>
    <li>Number of instances</li>
    <li>Log (Number of instances)</li>
	<li>Number of Features</li>
    <li>Log (Number of Features)</li>
    <li>Number of Classes</li>
    <li>Number of numerical Features</li>
    <li>Number of Categorical Features</li>
    <li>Ratio between Categorical & Numerical</li>
    <li>Number of Instances to Number of Features ratio</li>
    <li>Number of Missing Value</li>
    <li>Ratio of missing value</li>
	</ul>
</li>
<li>For Classes:
	<ul>
    <li>Class Entropy</li>
    <li>Classes Probabilities:(Minimum - Maximum - Standard deviation - Mean)</li>
	</ul>
</li>
<li>For Categorical Features:
	<ul>
    <li>Sum of all symbols</li>
    <li>Mean of all symbols</li>
    <li>Standard Deviation for all symbols</li>
	</ul>
</li>
<li>For Numerical Features:
	<ul>
    <li>Skewness for all Numerical features: (Max - Min - Standard Deviation - Mean)</li>
    <li>Kurtosis for all Numerical features: (Max - Min - Standard Deviation - Mean)</li>
	</ul>
</li>
</ul>

An important characteristic adds to the knowledge base is the accuracy against machine learning classifier.The used set of classifiers are:
<ul>
<li>Random Forest</li>
<li>Logistic Regression</li> 
<li>Decision Tree</li>
<li>Multilayer Perceptron</li>
<li>Linear SVC</li>
<li>Naïve Bayes</li>
<li>GBT</li> 
<li>LDA</li> 
<li>QDA </li>
</ul>
We have tested each dataset in the knowledge base against each algorithm and the the Algorithm with the best Accuracy with all Algorithms within 1 Standard Deviation rang will be marked as Good, other than that will be marked as bad
![Output](https://raw.githubusercontent.com/DataSystemsGroupUT/Distributed-SmartML/master/Images/kb1.png)
So, the Knowledge base contains 30 features used to represent each datasets and for each dataset we have 9 records, record per classifier, and label with two values (1 for good & 0 for bad) 
![Output](https://raw.githubusercontent.com/DataSystemsGroupUT/Distributed-SmartML/master/Images/kb2.png)

## Algorithm Selection
Since we have knowledge base contains the characteristics & the best classifier for a group of datasets, 
we can use machine learning to get the closest dataset in the knowledge base for an input dataset, then determine the expected best classifiers.

## Hyper parameter Optimization
We are using Hyperband in order to determine best hyperparameters quickly, 
hyperband based on Successive Halving algorithm which allocates exponentially more resources to more promising configurations, the algorithm do the following:
<ul>
<li>1)	Uniformly allocate a budget to a set of hyperparameter configurations</li>
<li>2)	Evaluate the performance of all configurations</li>
<li>3)	Throw out the worst half</li>
<li>4)	Repeat until one configuration remains</li>
</ul>
Hyperband consist of two loops:
<ul>
<li>the inner loop invokes SuccessiveHalving for fixed values of (hyperparameters configurations) and resources.</li>
<li>the outer loop iterates over different values of (hyperparameters configurations) and resources.</li>
</ul>
<br/>
we have used number of instances (data sampling) as the hyperband resource,so the maximum resources that can be allocated is 100% of data.<br/>

## Spark Implementation
We have used Apache Spark to distribute the process (Feature Extraction, Algorithm Selection and hyper parameter optimization).
We have also added two classifiers (LDA & QDA) to Spark ML to increase the number of the available classifiers.