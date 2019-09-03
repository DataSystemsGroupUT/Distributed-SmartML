
//package org.apache.spark.ml.tuning
package org.dsmartml

import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession

/**
  * Class Name: ClassifiersManager
  * Description: this calss responable for Managing all the avaialable classifiers and their Hyperparameters
  *
  * @constructor Create a Classifiers Manager represents all available classifiers with thier possible hyper parameters
  * @param spark the used spark session
  * @param nr_features the number of features in the dataset (used to build multilayer perceptron possible hyper parameter (layers)
  * @param nr_classes the number of classes in the dataset ((used to build multilayer perceptron possible hyper parameter (layers))
  * @param label the dataset label column name (default = y)
  * @param featuresCol the dataset features column name (default = features)
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/3/2019
  */
class ClassifiersManager(spark:SparkSession, nr_features:Int , nr_classes:Int, label:String = "y" , featuresCol:String = "features" ) {

  /**
    * Map contains classifier as a key and Array of parameter as a value
    */
  var ClassifierParamsMap =  Map[ String, Array[ParamMap]]()
  var ClassifierParamsMapIndexed =  Map[ String,Map[Int,ParamMap] ]()
  /**
    * Map contains classifier name as a key and Classifier object as a value
    */
  var ClassifiersMap = Map[String , Estimator[_]]()
  //implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(1234)))

  //--- Evaluator----------------------------------------------------
  /**
    * Multiclass Classification Evaluator object with evaluation Metric set to accuracy
    * used to evalaute each classifier
    */
  val evaluator = new MulticlassClassificationEvaluator()
    .setLabelCol(label)
    .setPredictionCol("prediction")
    .setMetricName("accuracy")

  //---RandomForest---------------------------------------------------
  /**
    * Random Forest Classifier Object with label and feature column configured
    */
  val rf = new RandomForestClassifier()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)
  /**
    * Parameter Grid builder Object for Random Forest possible hyper parameters values (ex:numTrees, maxDepth, maxBins,...)
    */
  var rf_paramGrid = new ParamGridBuilder()
    .addGrid(rf.numTrees, ClassifiersManager.numTrees)
    .addGrid(rf.maxDepth, ClassifiersManager.maxDepth)
    .addGrid(rf.maxBins, ClassifiersManager.maxBins)
    .addGrid(rf.minInfoGain, ClassifiersManager.minInfoGain)
    .addGrid(rf.minInstancesPerNode, ClassifiersManager.minInstancesPerNode )
    .addGrid(rf.impurity, ClassifiersManager.impurity)
    .build()

  // i added this ugly loops, to give each combination of hyperparameters and Index
  // that are fixied from run to run.
  //so, if i used the same seed to select random parameter for hyperband, i obtained the same result
  var rf_counter = 0
  var rf_indexedParamMapArr = Map[Int,ParamMap]()
  for( i <-ClassifiersManager.numTrees )
    {
      for(j <- ClassifiersManager.maxDepth)
        {
          for (k <- ClassifiersManager.maxBins)
            {
              for ( l <- ClassifiersManager.minInfoGain)
              {
                for(m <- ClassifiersManager.impurity)
                {
                  //for (n <- ClassifiersManager.minInstancesPerNode) {
                    var pm = new ParamMap()
                    pm.put(rf.numTrees, i)
                    pm.put(rf.maxDepth, j)
                    pm.put(rf.maxBins, k)
                    pm.put(rf.minInfoGain, l)
                    pm.put(rf.impurity, m)
                    //pm.put(rf.minInstancesPerNode, n )
                    rf_indexedParamMapArr += (rf_counter -> pm)
                    rf_counter = rf_counter + 1
                 // }
                }
              }
            }
        }
    }
  ClassifierParamsMapIndexed += ("RandomForestClassifier" -> rf_indexedParamMapArr)
  ClassifiersMap +=("RandomForestClassifier" -> rf)
  ClassifierParamsMap +=("RandomForestClassifier" -> rf_paramGrid)


  def setSeed(seed: Int = 0) = {
    new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
  }

  def getRandomParameters( classifier:String, Count:Int): Array[ParamMap] = {

    val randomGrid = new RandomGridBuilder(Count)

    classifier match {
      case "RandomForestClassifier" =>
        randomGrid.addDistr(rf.numTrees, Poisson(60)(setSeed(1234))) //ClassifiersManager.numTrees)
        randomGrid.addDistr(rf.maxDepth,  Poisson(12)(setSeed(1234)) ) //ClassifiersManager.maxDepth)
        randomGrid.addDistr(rf.maxBins,  ClassifiersManager.maxBins)
        randomGrid.addDistr(rf.minInfoGain, ClassifiersManager.minInfoGain) // Gamma(0.5,0.1))
        randomGrid.addDistr(rf.minInstancesPerNode, ClassifiersManager.minInstancesPerNode )
        randomGrid.addDistr(rf.impurity, ClassifiersManager.impurity)

      case  "LogisticRegression" =>
        randomGrid.addDistr(lr.fitIntercept, ClassifiersManager.fitIntercept)
        randomGrid.addDistr(lr.maxIter, ClassifiersManager.maxIter)
        randomGrid.addDistr(lr.regParam, Gamma(0.5,0.1 )(setSeed(1234))) //ClassifiersManager.regParam)
        randomGrid.addDistr(lr.elasticNetParam, Gamma(0.5,0.1)(setSeed(1234))) //ClassifiersManager.elasticNetParam)
        randomGrid.addDistr(lr.standardization, ClassifiersManager.standardization)
        randomGrid.addDistr(lr.tol, ClassifiersManager.tol)

      case "DecisionTreeClassifier" =>
        randomGrid.addDistr(dt.maxDepth,  Poisson(9)(setSeed(1234)) ) // ClassifiersManager.maxDepth)
        randomGrid.addDistr(dt.maxBins, ClassifiersManager.maxBins)
        randomGrid.addDistr(dt.minInfoGain, Gamma(1.5,0.1)(setSeed(1234)) ) //ClassifiersManager.minInfoGain)
        randomGrid.addDistr(dt.minInstancesPerNode, ClassifiersManager.minInstancesPerNode )
        randomGrid.addDistr(dt.impurity, ClassifiersManager.impurity)

      case "MultilayerPerceptronClassifier" =>
        randomGrid.addDistr(mpr.layers, Array(Array(nr_features,2,nr_classes),Array(nr_features,3,nr_classes),Array(nr_features,4,nr_classes)  ,
                            Array(nr_features, 5,  nr_classes)))
        randomGrid.addDistr(mpr.maxIter ,ClassifiersManager.maxIter )

      case "LinearSVC" =>
        randomGrid.addDistr(lsvc.maxIter ,Poisson(30)(setSeed(1234)))// ClassifiersManager.maxIter)
        randomGrid.addDistr(lsvc.regParam, Gamma(0.5,0.1)(setSeed(1234)))//ClassifiersManager.regParam)

      case "NaiveBayes" =>
        randomGrid.addDistr(nb.smoothing, ClassifiersManager.smoothing)

      case "GBTClassifier" =>
        randomGrid.addDistr(gbt.maxDepth, Poisson(10)(setSeed(1234))) //ClassifiersManager.maxDepth)
        randomGrid.addDistr(gbt.maxBins, Array(16,32,64,128,256,512,1024))
        randomGrid.addDistr(gbt.minInfoGain, Gamma(1.5,0.1)(setSeed(1234))) //ClassifiersManager.minInfoGain)
        randomGrid.addDistr(gbt.minInstancesPerNode, ClassifiersManager.minInstancesPerNode )

      case "LDA" =>
        randomGrid.addDistr(lda.scaledData ,Array(false,true) )

      case "QDA" =>
        randomGrid.addDistr(qda.scaledData ,Array(false,true) )

      //.addGrid(gbt.impurity, Array("gini" , "entropy"))
            /*randomGrid.addDistr (lr.regParam, Gamma (0.5, 0.1) )
            randomGrid.addDistr (lr.elasticNetParam, Gamma (0.5, 0.1) )
            randomGrid.addDistr (lr.threshold, Gaussian (0.5, 0.05) )
            randomGrid.addDistr (lr.standardization, Array (true, false) )*/

    }
    val rslt = randomGrid.build()
    //rslt.toList.foreach(println)
    return rslt
  }

  def getRandomParametersIndexed(classifier:String, Count:Int): Map[Int , ParamMap] =
  {
    var SelectedParamsMapIndexed =  Map[Int,ParamMap]()
    var iter = 1
    var arr= getRandomParameters(classifier, Count)
    for (x <- arr) {
      SelectedParamsMapIndexed += (iter -> x)
      iter = iter + 1
    }
    /*
    val ParamStep = 5
    var RemainingParamCount = Count
    var curParamNumber = 0
    var parametercount = 0
    while ( RemainingParamCount > 0)
    {
      if (RemainingParamCount > ParamStep)
      {
      parametercount = ParamStep
      RemainingParamCount = RemainingParamCount - ParamStep
      }
      else
      {
      parametercount = RemainingParamCount
      RemainingParamCount = 0
      }
      var arr= getRandomParameters(classifier, parametercount)
      //var arr = getRandomParameters(classifier, Count)

      for (x <- arr) {
        SelectedParamsMapIndexed += (iter -> x)
        iter = iter + 1
      }
    }

  */
    //SelectedParamsMapIndexed.map( x => println(x._2))
    return SelectedParamsMapIndexed
  }
  //---Logistic Regression--------------------------------------------
  /**
    * Logistic Regression Classifier Object with label and feature column configured
    */
  val lr = new LogisticRegression()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)
  /**
    * Parameter Grid builder Oject for Logistic Regression possible hyper parameters values (ex:elasticNetParam, maxIter, regParam,...)
    */
  var lr_paramGrid = new ParamGridBuilder()

    .addGrid(lr.fitIntercept, ClassifiersManager.fitIntercept)
    .addGrid(lr.maxIter, ClassifiersManager.maxIter)
    .addGrid(lr.regParam, ClassifiersManager.regParam)
    .addGrid(lr.elasticNetParam, ClassifiersManager.elasticNetParam)
    .addGrid(lr.standardization, ClassifiersManager.standardization)
    .addGrid(lr.tol, ClassifiersManager.tol)
    .build()
  var lr_counter = 0
  var lr_indexedParamMapArr = Map[Int,ParamMap]()
  for( i <-ClassifiersManager.fitIntercept )
  {
    for(j <- ClassifiersManager.maxIter)
    {
      for (k <- ClassifiersManager.regParam)
      {
        for ( l <- ClassifiersManager.elasticNetParam)
        {
          for(m <- ClassifiersManager.standardization) {
            var pm = new ParamMap()
            pm.put(lr.fitIntercept, i)
            pm.put(lr.maxIter, j)
            pm.put(lr.regParam, k)
            pm.put(lr.elasticNetParam, l)
            pm.put(lr.standardization, m)
            lr_indexedParamMapArr += (lr_counter -> pm)
            lr_counter = lr_counter + 1
          }
        }
      }
    }
  }
  ClassifierParamsMapIndexed += ("LogisticRegression" -> lr_indexedParamMapArr)
  ClassifiersMap += ( "LogisticRegression" -> lr)
  ClassifierParamsMap +=("LogisticRegression" -> lr_paramGrid)

  //---Decision Tree--------------------------------------------------
  /**
    * Decision Tree Classifier Object with label and feature column configured
    */
  val dt = new DecisionTreeClassifier()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)
  /**
    * Parameter Grid builder Object for possible ecision Tree Classifier hyper parameters values (ex:impurity, maxDepth, maxBins,...)
    */
  var dt_paramGrid = new ParamGridBuilder()
    .addGrid(dt.maxDepth, ClassifiersManager.maxDepth)
    .addGrid(dt.maxBins, ClassifiersManager.maxBins)
    .addGrid(dt.minInfoGain, ClassifiersManager.minInfoGain)
    //.addGrid(dt.minInstancesPerNode, Array(1,5,10,100) )
    .addGrid(dt.impurity, ClassifiersManager.impurity)
    .build()
  var dt_counter = 0
  var dt_indexedParamMapArr = Map[Int,ParamMap]()
  for( i <-ClassifiersManager.maxDepth )
  {
    for(j <- ClassifiersManager.maxBins)
    {
      for (k <- ClassifiersManager.minInfoGain)
      {
        for ( l <- ClassifiersManager.impurity)
        {
            var pm = new ParamMap()
            pm.put(dt.maxDepth, i)
            pm.put(dt.maxBins, j)
            pm.put(dt.minInfoGain, k)
            pm.put(dt.impurity, l)
            dt_indexedParamMapArr += (dt_counter -> pm)
            dt_counter = dt_counter + 1
        }
      }
    }
  }
  ClassifierParamsMapIndexed += ("DecisionTreeClassifier" -> dt_indexedParamMapArr)
  ClassifiersMap += ( "DecisionTreeClassifier" -> dt)
  ClassifierParamsMap +=("DecisionTreeClassifier" -> dt_paramGrid)

  //---Multilayer Perceptron-------------------------------------------
  /**
    * Multilayer Perceptron Classifier Object with label and feature column configured
    */
  val mpr = new MultilayerPerceptronClassifier()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)
  /**
    * Parameter Grid builder Oject for possible Multilayer Perceptron Classifier hyper parameters values (ex:layers, maxIter,...)
    */
  var mpr_paramGrid = new ParamGridBuilder()
    .addGrid(mpr.layers, Array(Array(nr_features,2,nr_classes),Array(nr_features,3,nr_classes),Array(nr_features,4,nr_classes)  ,
      Array(nr_features, 5,  nr_classes)
      //Array(nr_features, 6,  nr_classes)  ,
      //Array(nr_features, 7,  nr_classes)  ,
      //Array(nr_features, 8,  nr_classes)  ,
      //Array(nr_features, 10,  nr_classes)  ,
      //Array(nr_features, 15, nr_classes)
    ))
    //.addGrid(mpr.layers, Array( Array(10, 3, 12) ))
    .addGrid(mpr.maxIter ,ClassifiersManager.maxIter )
    .build()

  var mpr_counter = 0
  var mpr_indexedParamMapArr = Map[Int,ParamMap]()
  for( i <-Array(Array(nr_features,2,nr_classes),Array(nr_features,3,nr_classes),Array(nr_features,4,nr_classes)  ,
    Array(nr_features, 5,  nr_classes)
    //Array(nr_features, 6,  nr_classes)  ,
    //Array(nr_features, 7,  nr_classes)  ,
    //Array(nr_features, 8,  nr_classes)  ,
    //Array(nr_features, 10,  nr_classes)  ,
    //Array(nr_features, 15, nr_classes)
  ) )
  {
    for(j <- ClassifiersManager.maxIter)
    {
      var pm = new ParamMap()
      pm.put(mpr.layers, i)
       pm.put(mpr.maxIter, j)
      mpr_indexedParamMapArr += (mpr_counter -> pm)
      mpr_counter = mpr_counter + 1
    }
  }
  ClassifierParamsMapIndexed += ("MultilayerPerceptronClassifier" -> mpr_indexedParamMapArr)
  ClassifiersMap += ("MultilayerPerceptronClassifier" -> mpr)
  ClassifierParamsMap +=("MultilayerPerceptronClassifier" -> mpr_paramGrid)

  //---Linear SVC------------------------------------------------------
  /**
    * Linear SVC Classifier Object with label and feature column configured
    */
  val lsvc = new LinearSVC()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)
  /**
    * Parameter Grid builder Oject for possible Linear SVC Classifier hyper parameters values (ex:regParam, maxIter,...)
    */
  var lsvc_paramGrid = new ParamGridBuilder()
    .addGrid(lsvc.maxIter ,ClassifiersManager.maxIter)
    .addGrid(lsvc.regParam, ClassifiersManager.regParam)
    .build()

  var lsvc_counter = 0
  var lsvc_indexedParamMapArr = Map[Int,ParamMap]()
  for( i <-ClassifiersManager.maxIter )
  {
    for(j <- ClassifiersManager.regParam)
    {
         var pm = new ParamMap()
          pm.put(lsvc.maxIter, i)
          pm.put(lsvc.regParam, j)
      lsvc_indexedParamMapArr += (dt_counter -> pm)
      lsvc_counter = dt_counter + 1
    }
  }
  ClassifierParamsMapIndexed += ("LinearSVC" -> lsvc_indexedParamMapArr)
  ClassifiersMap += ( "LinearSVC" -> lsvc)
  ClassifierParamsMap +=("LinearSVC" -> lsvc_paramGrid)

  //---NaiveBayes------------------------------------------------------
  /**
    * Naive Bayes Classifier Object with label and feature column configured
    */
  val nb = new NaiveBayes()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)
  /**
    * Parameter Grid builder Object for possible Naive Bayes Classifier hyper parameters values (ex:modelType, smoothing,...)
    */
  var nb_paramGrid = new ParamGridBuilder()
    //.addGrid(nb.modelType ,Array("multinomial" , "bernoulli") )
    .addGrid(nb.smoothing, ClassifiersManager.smoothing)
    .build()

  var nb_counter = 0
  var nb_indexedParamMapArr = Map[Int,ParamMap]()
  for( i <-ClassifiersManager.smoothing )
  {
      var pm = new ParamMap()
      pm.put(nb.smoothing, i)
      nb_indexedParamMapArr += (nb_counter -> pm)
      nb_counter = nb_counter + 1
  }
  ClassifierParamsMapIndexed += ("NaiveBayes" -> nb_indexedParamMapArr)
  ClassifiersMap += ( "NaiveBayes" -> nb)
  ClassifierParamsMap +=("NaiveBayes" -> nb_paramGrid)

  //---GBTClassifier---------------------------------------------------
  /**
    * GBT Classifier Object with label and feature column configured
    */
  val gbt = new GBTClassifier()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)
  /**
    * Parameter Grid builder Object for possible GBT Classifier hyper parameters values (ex:maxDepth, maxBins,...)
    */
  var gbt_paramGrid = new ParamGridBuilder()
    .addGrid(gbt.maxDepth, ClassifiersManager.maxDepth)
    //.addGrid(gbt.maxBins, Array(16,32,64,128,256,512,1024))
    .addGrid(gbt.minInfoGain, ClassifiersManager.minInfoGain)
    .addGrid(gbt.minInstancesPerNode, ClassifiersManager.minInstancesPerNode )
    //.addGrid(gbt.impurity, Array("gini" , "entropy"))
    .build()

  var gbt_counter = 0
  var gbt_indexedParamMapArr = Map[Int,ParamMap]()
  for( i <-ClassifiersManager.maxDepth )
  {
    for(j <- ClassifiersManager.minInfoGain)
    {
      for (k <- ClassifiersManager.minInstancesPerNode)
      {
        var pm = new ParamMap()
        pm.put(gbt.maxDepth, i)
        pm.put(gbt.minInfoGain, j)
        pm.put(gbt.minInstancesPerNode, k)
        gbt_indexedParamMapArr += (gbt_counter -> pm)
        gbt_counter = gbt_counter + 1
      }
    }
  }
  ClassifierParamsMapIndexed += ("GBTClassifier" -> gbt_indexedParamMapArr)
  ClassifiersMap += ( "GBTClassifier" -> gbt)
  ClassifierParamsMap +=("GBTClassifier" -> gbt_paramGrid)

  //---LDA------------------------------------------------------------
  //val lda = new org.apache.spark.ml.classification.LDA(spark.sparkContext)
  /**
    * LDA Classifier Object with label and feature column configured
    */
  val lda = new org.apache.spark.ml.classification.LDA()
  lda.sc = spark.sparkContext
  lda.setLabelCol(label)
  lda.setFeaturesCol(featuresCol)
  lda.setPredictionCol("prediction")
  lda.setProbabilityCol("Probability")
  lda.setRawPredictionCol("RawPrediction")

  LDAUtil.sc = spark.sparkContext
  /**
    * Parameter Grid builder Oject for LDA possible hyper parameters values (ex:scaledData,...)
    */
  var lda_paramGrid = new ParamGridBuilder()
    .addGrid(lda.scaledData ,Array(false) )
    .build()
  ClassifierParamsMapIndexed += ("LDA" -> Map(1 -> new ParamMap().put(lda.scaledData ,false)  ))
  ClassifiersMap += ( "LDA" -> lda)
  ClassifierParamsMap +=("LDA" -> lda_paramGrid)


  //---QDA------------------------------------------------------------
  /**
    * QDA Classifier Object with label and feature column configured
    */
  val qda = new org.apache.spark.ml.classification.QDA(spark.sparkContext)
  qda.setLabelCol(label)
  qda.setFeaturesCol(featuresCol)
  qda.setPredictionCol("prediction")
  qda.setProbabilityCol("Probability")
  qda.setRawPredictionCol("RawPrediction")
  /**
    * Parameter Grid builder Object for QDA possible hyper parameter svalues (ex:scaledData,...)
    */
  var qda_paramGrid = new ParamGridBuilder()
    .addGrid(qda.scaledData ,Array(true,false) )
    .build()
  ClassifierParamsMapIndexed += ("QDA" -> Map(1 -> new ParamMap().put(lda.scaledData ,false)  ))
  ClassifiersMap += ( "QDA" -> qda)
  ClassifierParamsMap +=("QDA" -> qda_paramGrid)


}

object ClassifiersManager {
  /**
    * List contains the names of all available classifiers
    */
  var classifiersLsit = List("RandomForestClassifier" , "LogisticRegression" , "DecisionTreeClassifier" ,
    "MultilayerPerceptronClassifier" , "LinearSVC" ,"NaiveBayes" , "GBTClassifier" ,
    "LDA" , "QDA" )

  /**
    * this list represent the Order of the Algorithm evaluation
    */
  val OrderList = List(0,6,1,2,3,4,5,7,8)


  val numTrees    = Array(50,60,70)
  val maxDepth    = Array(6,7,8,9,10,11,13,15,17,19)
  val minInfoGain = Array(0.001, 0.01 , 0.1 )
  val impurity    = Array("gini" )//, "entropy")
  val maxBins =  Array(96)

  val maxIter = Array(50,60,70)
  val fitIntercept =  Array(true)
  val regParam = Array(0.001,0.01,0.1)
  val elasticNetParam = Array(0.05,0.1,0.5)
  val standardization = Array(true)
  val tol= Array(1E-6)
  val smoothing =  Array(1.0 , 0.5)
  val minInstancesPerNode= Array(1, 5, 10)


  def SortedSelectedClassifiers( lst:Array[Double]) : List[Int] =
  {
    var rslt= List[Int]()
    for( i <- OrderList ) {
        if ( lst.contains(i))
          rslt = i :: rslt
      }

  return rslt.reverse
  }
}
