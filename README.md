## Introduction ##
Machine learning model building is a highly iterative exploratory process where most scientists work hard to find the best model or algorithm that meets their performance requirement.
In practice, there is no one-model-fits-all solutions, thus, there is no single model or algorithm that can handle all data set varieties and changes in data that may occur over time. 
 
Also each machine learning algorithm require user defined inputs to achieve a balance between accuracy and generalizability (which called hyperparameters). 
So the process of selecting the Algorithm and its hyperparameters usually require a lot of trails until we find it. In case of big datasets this process will need a lot of resources (Time, Hardware, effort, …).


## Distributed Smart-ML Library

### The Main Idea
The library receive dataset as an input and produce an optimized model as an output.
The library extracts some characteristics of the datasets and use an internal knowledgebase to determine the best algorithm, then use a hyperband method to find the best hyper parameters for the selected algorithm.
 
Finally, the datasets characteristics (meta-data) and its selected algorithm added as a feedback to the knowledgebase.
![Main Idea](https://github.com/DataSystemsGroupUT/Distributed-SmartML/blob/master/Images/MainIdea.png)

### Datasets Meta Extraction & our Knowledgebase
For each dataset, a set of meta-data extracted that represent statistics and description for this dataset, the extracted meta data can be grouped as following:
- For All the Dataset
    - Number of instances
    - Log (Number of instances)
    - Number of Features
    - Log (Number of Features)
    - Number of Classes
    - Number of numerical Features
    - Number of Categorical Features
    - Ratio between Categorical & Numerical
    - Number of Instances to Number of Features ratio
    - Number of Missing Value
    - atio of missing value
- For Classes
    - Class Entropy
    - Classes Probabilities:(Minimum - Maximum - Standard deviation - Mean)
- For Categorical Features
    - Sum of all symbols
    - Mean of all symbols
    - Standard Deviation for all symbols
- For Numerical Features
    - Skewness for all Numerical features: (Max - Min - Standard Deviation - Mean)
    - Kurtosis for all Numerical features: (Max - Min - Standard Deviation - Mean)

An important characteristic adds to the knowledge base is the accuracy against machine learning classifier.The used set of classifiers are:
-	Random Forest 
-	Logistic Regression 
-	Decision Tree
-	Multilayer Perceptron 
-	Linear SVC 
-	Naïve Bayes 
-	GBT 
-	LDA 
-	QDA 

### Algorithm Selection
Since we have knowledge base contains the characteristics & the best classifier for a group of datasets, we can use machine learning to get the closest dataset in the knowledge base for an input dataset, then determine the expected best classifiers.

### Hyper parameter Optimization
We are using Hyperband in order to determine best hyperparameters quickly, hyperband based on Successive Halving algorithm which allocates exponentially more resources to more promising configurations, the algorithm do the following:
1)	Uniformly allocate a budget to a set of hyperparameter configurations
2)	Evaluate the performance of all configurations
3)	Throw out the worst half
4)	Repeat until one configuration remains
   
Hyperband consist of two loops:
- the inner loop invokes SuccessiveHalving for fixed values of (hyperparameters configurations) and resources.
- the outer loop iterates over different values of (hyperparameters configurations) and resources.

we have used number of instances (data sampling) as the hyperband resource,so the maximum resources that can be allocated is 100% of data.

### Spark Implementation
We have used Apache Spark to distribute the process (Feature Extraction, Algorithm Selection and hyper parameter optimization).
We have also added two classifiers (LDA & QDA) to Spark ML to increase the number of the available classifiers.

## Installation
The library published on Maven public repo and it can be referenced easily as shown below:


>**Maven**
```sh
<dependency>
	<groupId>com.github.DataSystemsGroupUT</groupId>
	<artifactId>D-SmartML</artifactId>
	<version>0.1</version>
</dependency>
```

 <br />
 
 >**Scala SBT**
```sh
 libraryDependencies += "com.github.DataSystemsGroupUT" % "D-SmartML" % "0.1"
 ```
 
  <br />
 
 >**Groovy Grape**
```sh
@Grapes(
@Grab(group='com.github.DataSystemsGroupUT', module='D- SmartML', version='0.1')
)
 ```
 <br />
 
 >**Gradle/Grails**
```sh
 compile 'com.github.ahmed-eissa:DSmartML:0.2.4'
 ```

## Library Parameters

| Parameter| Description | Data Type | Default Value |
| ------ | ------ |------ |------ |
| **eta**| an input that controls the proportion of configurations discarded in each round of  SuccessiveHalving (in hyperband) | Integer | 5 |
|**Max Data Percentage**|the maximum amount of resource that can be allocated to a single configuration|Integer| 100 |
|**Parallelism**|the maximum amount of resource that can be allocated to a single configuration (models will only be run in parallel if there are enough resources available in the cluster. Otherwise, models will be queued in the Spark scheduler and have to wait for the current jobs to complete before being run.)|Integer|1|
|**Try N Classifier**|Maximum Number of Algorithms should be checked (out of the best algorithms based on the kB)|Integer|2|
|**Max Time**|Maximum Time allowed for hyper parameter optimization (per each Algorithm) to get the best hyperparameter values  (in Seconds)| Integer|1800|
|**HP Optimizer**|Hyper parameters optimizer (1: Random Search or 2: Hyperband)|Integer|2|
|**Convert To Vector Assembly**|If the input dataset features need to be converted to Vector or not|Boolean|false|




## Examples
This is a complete example of using the D-Smart ML library to get the best model for a given dataset (csv file), the example does the following:
- Create Spark Session
- Load dataset from CSV file (as DataFrame)
- Create D-Smart ML Model Selector object and call “getBestModel” function

![Example](https://github.com/DataSystemsGroupUT/Distributed-SmartML/blob/master/Images/Ex.png)

	
## Library Output
The below images, is a sample of the  library output

![Output](https://github.com/DataSystemsGroupUT/Distributed-SmartML/blob/master/Images/Output_.png)










