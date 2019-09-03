//package org.apache.spark.ml.tuning
package org.dsmartml
/**
  * Class Name: DatasetMetadata
  * Description: this class represent dataset metadata (Statistics & Classification metadata)
  * @constructor Create an empty object for dataset metadata
  */
class DatasetMetadata extends java.io.Serializable{

  /**
    * the name of the dataset
    */
  var datasetname = ""

  // Dataset Statistical Features
  //=========================================================================
  /**
    * the number of instances in the dataset (number of rows)
    */
  var nr_instances:Long = 0
  /**
    * Mathematic Log for the number of instances in the dataset (number of rows)
    */
  var log_nr_instances = 0.0
  /**
    * the number of features in the dataset (number of columns)
    */
  var nr_features = 0
  /**
    * Mathematic Log for the number of features in the dataset (number of columns)
    */
  var log_nr_features = 0.0
  /**
    * Number of classes in the Dataset
    */
  var nr_classes = 0
  /**
    * the number of numerical features in the dataset
    */
  var nr_numerical_features = 0
  /**
    * the number of categorical features in the dataset
    */
  var nr_categorical_features = 0
  /**
    * the ratio between numerical and categorical features in the dataset
    */
  var ratio_num_cat =0.0
  /**
    * the class entropy in the dataset
    */
  var class_entropy = 0.0
  /**
    * the sum of the number of missing values in all the features in the dataset
    */
  var missing_val :Long= 0
  /**
    * the ration between avaialable and missing values in all the features in the dataset
    */
  var ratio_missing_val = 0.0
  /**
    * the maximum propability for dataset classes (ex: if we have three classes A= 30%, B=20%, C=50% then max_prob = 50)
    */
  var max_prob = 0.0
  /**
    * the minimum propability for dataset classes (ex: if we have three classes A= 30%, B=20%, C=50% then min_prob = 20)
    */
  var min_prob = 0.0
  /**
    * the maximum propability for dataset classes (ex: if we have three classes A= 30%, B=20%, C=50% then mean_prob = 30+50+20/3)
    */
  var mean_prob = 0.0
  /**
    * the standard deviation of the classess probabilities
    */
  var std_dev = 0.0
  /**
    * the ratio between columns and rows (features & instances)
    */
  var dataset_ratio = 0.0
  /**
    * for categorical features, the sum of all indecies
    */
  var symbols_sum = 0.0
  /**
    * for categorical features, the mean of all indecies
    */
  var symbols_mean = 0.0
  /**
    * for categorical features, the standard deviation of all indecies
    */
  var symbols_std_dev = 0.0
  /**
    * for numerical features, the min skew of all features
    */
  var skew_min = 0.0
  /**
    * for numerical features, the max skew of all features
    */
  var skew_max = 0.0
  /**
    * for numerical features, the mean skew of all features
    */
  var skew_mean = 0.0
  /**
    * for numerical features, the standard deviation skew of all features
    */
  var skew_std_dev = 0.0
  /**
    * for numerical features, the min kurtosis of all features
    */
  var kurtosis_min = 0.0
  /**
    * for numerical features, the max kurtosis of all features
    */
  var kurtosis_max = 0.0
  /**
    * for numerical features, the mean kurtosis of all features
    */
  var kurtosis_mean = 0.0
  /**
    * for numerical features, the standard deviation for kurtosis of all features
    */
  var kurtosis_std_dev = 0.0
  /**
    * if any features has negative values
    */
  var hasNegativeFeatures = false

  //Dataset Classification Features
  //=========================================================================
  /**
    * the accuracy of algorithm (RandomForestClassifier) on this dataset using default hyperparameters
    */
  var RandomForestClassifier_Accuracy = 0.0
  /**
    * the accuracy of algorithm (LogisticRegression) on this dataset using default hyperparameters
    */
  var LogisticRegression_Accuracy = 0.0
  /**
    * the accuracy of algorithm (DecisionTreeClassifier) on this dataset using default hyperparameters
    */
  var DecisionTreeClassifier_Accuracy = 0.0
  /**
    * the accuracy of algorithm (MultilayerPerceptronClassifier) on this dataset using default hyperparameters
    */
  var MultilayerPerceptronClassifier_Accuracy = 0.0
  /**
    * the accuracy of algorithm (LinearSVC) on this dataset using default hyperparameters
    */
  var LinearSVC_Accuracy= 0.0
  /**
    * the accuracy of algorithm (NaiveBayes) on this dataset using default hyperparameters
    */
  var NaiveBayes_Accuracy = 0.0
  /**
    * the accuracy of algorithm (GBTClassifier) on this dataset using default hyperparameters
    */
  var GBTClassifier_Accuracy = 0.0
  /**
    * the accuracy of algorithm (LDA) on this dataset using default hyperparameters
    */
  var LDA_Accuracy = 0.0
  /**
    * the accuracy of algorithm (QDA) on this dataset using default hyperparameters
    */
  var QDA_Accuracy = 0.0


  /**
    * the Order of the algorithm (RandomForestClassifier) based on its accuracy (0 is the best algorithm)
    */
  var RandomForestClassifier_Order = 0
  /**
    * the Order of the algorithm (LogisticRegression) based on its accuracy (0 is the best algorithm)
    */
  var LogisticRegression_Order = 0
  /**
    * the Order of the algorithm (DecisionTreeClassifier) based on its accuracy (0 is the best algorithm)
    */
  var DecisionTreeClassifier_Order = 0
  /**
    * the Order of the algorithm (MultilayerPerceptronClassifier) based on its accuracy (0 is the best algorithm)
    */
  var MultilayerPerceptronClassifier_Order = 0
  /**
    * the Order of the algorithm (LinearSVC) based on its accuracy (0 is the best algorithm)
    */
  var LinearSVC_Order= 0
  /**
    * the Order of the algorithm (NaiveBayes) based on its accuracy (0 is the best algorithm)
    */
  var NaiveBayes_Order = 0
  /**
    * the Order of the algorithm (GBTClassifier) based on its accuracy (0 is the best algorithm)
    */
  var GBTClassifier_Order = 0
  /**
    * the Order of the algorithm (LDA) based on its accuracy (0 is the best algorithm)
    */
  var LDA_Order = 0
  /**
    * the Order of the algorithm (QDA) based on its accuracy (0 is the best algorithm)
    */
  var QDA_Order = 0

  /**
    * Map contains the Alorithm name as key and its Accuracy as value
    */
  var accuracyMap = Map[String, Double]()

  /**
    * Map contains the Alorithm name as key and its Order as value
    */
  var accOrderMap = Map[String, Integer]()

  //Dataset Best Classifier
  //=========================================================================
  /**
    * the name of the best algorithm based on the accuracy
    */
  var BestAlgorithm = ""

  /**
    * the Value of the accuracy of the best algorithm
    */
  var BestAlgorithm_Accuracy = 0.0


  //Dataset Best Classifier Threshold
  //=========================================================================
  var Acc_Std_threshold = 0.0
  var Acc_Ord_threshold = 0.0
  var Acc_Std_Ord_threshold = 0.0
  var topn = 3

}
