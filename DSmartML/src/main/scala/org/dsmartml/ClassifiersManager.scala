package org.dsmartml

import breeze.stats.distributions._
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.ml.Estimator
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.ParamGridBuilder
import org.apache.spark.sql.SparkSession

import scala.collection.mutable.ListBuffer

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
class ClassifiersManager(spark:SparkSession, nr_features:Int , nr_classes:Int, label:String = "y" , featuresCol:String = "features" , seed:Int = 1234 ) {

  /**
    * Map contains classifier name as a key and Classifier object as a value
    */
  var ClassifiersMap = Map[String , Estimator[_]]()
  var ClassifierParamsMap =  Map[ String, Array[ParamMap]]()

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

  var rf_paramGrid = new ParamGridBuilder()
    .addGrid(rf.numTrees, ClassifiersManager.numTrees)
    .addGrid(rf.maxDepth, ClassifiersManager.maxDepth)
    .addGrid(rf.maxBins, ClassifiersManager.maxBins)
    .addGrid(rf.minInfoGain, ClassifiersManager.minInfoGain)
    .addGrid(rf.minInstancesPerNode, ClassifiersManager.minInstancesPerNode )
    .addGrid(rf.impurity, ClassifiersManager.impurity)
    .build()

  ClassifiersMap +=("RandomForestClassifier" -> rf)
  ClassifierParamsMap +=("RandomForestClassifier" -> rf_paramGrid)


  def setSeed(seed: Int = 0) = {
    new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
  }

  //---Logistic Regression--------------------------------------------
  /**
    * Logistic Regression Classifier Object with label and feature column configured
    */
  val lr = new LogisticRegression()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)

  var lr_paramGrid = new ParamGridBuilder()
    .addGrid(lr.fitIntercept, ClassifiersManager.fitIntercept)
    .addGrid(lr.maxIter, ClassifiersManager.maxIter)
    .addGrid(lr.regParam, ClassifiersManager.regParam)
    .addGrid(lr.elasticNetParam, ClassifiersManager.elasticNetParam)
    .addGrid(lr.standardization, ClassifiersManager.standardization)
    .addGrid(lr.tol, ClassifiersManager.tol)
    .build()

  ClassifiersMap += ( "LogisticRegression" -> lr)
  ClassifierParamsMap +=("LogisticRegression" -> lr_paramGrid)

  //---Decision Tree--------------------------------------------------
  /**
    * Decision Tree Classifier Object with label and feature column configured
    */
  val dt = new DecisionTreeClassifier()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)

  var dt_paramGrid = new ParamGridBuilder()
    .addGrid(dt.maxDepth, ClassifiersManager.maxDepth)
    .addGrid(dt.maxBins, ClassifiersManager.maxBins)
    .addGrid(dt.minInfoGain, ClassifiersManager.minInfoGain)
    //.addGrid(dt.minInstancesPerNode, Array(1,5,10,100) )
    .addGrid(dt.impurity, ClassifiersManager.impurity)
    .build()

  ClassifiersMap += ( "DecisionTreeClassifier" -> dt)
  ClassifierParamsMap +=("DecisionTreeClassifier" -> dt_paramGrid)

  //---Multilayer Perceptron-------------------------------------------
  /**
    * Multilayer Perceptron Classifier Object with label and feature column configured
    */
  val mpr = new MultilayerPerceptronClassifier()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)

  var mpr_paramGrid = new ParamGridBuilder()
    .addGrid(mpr.layers, Array(
                         Array(nr_features,2,nr_classes),
                         Array(nr_features,3,nr_classes),
                         Array(nr_features,4,nr_classes),
                         Array(nr_features,5,nr_classes)
    ))
    .addGrid(mpr.maxIter ,ClassifiersManager.maxIter )
    .build()

  ClassifiersMap += ("MultilayerPerceptronClassifier" -> mpr)
  ClassifierParamsMap +=("MultilayerPerceptronClassifier" -> mpr_paramGrid)

  //---Linear SVC------------------------------------------------------
  /**
    * Linear SVC Classifier Object with label and feature column configured
    */
  val lsvc = new LinearSVC()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)

  var lsvc_paramGrid = new ParamGridBuilder()
    .addGrid(lsvc.maxIter ,ClassifiersManager.maxIter)
    .addGrid(lsvc.regParam, ClassifiersManager.regParam)
    .build()

  ClassifiersMap += ( "LinearSVC" -> lsvc)
  ClassifierParamsMap +=("LinearSVC" -> lsvc_paramGrid)

  //---NaiveBayes------------------------------------------------------
  /**
    * Naive Bayes Classifier Object with label and feature column configured
    */
  val nb = new NaiveBayes()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)

  var nb_paramGrid = new ParamGridBuilder()
    .addGrid(nb.smoothing, ClassifiersManager.smoothing)
    .build()

  ClassifiersMap += ( "NaiveBayes" -> nb)
  ClassifierParamsMap +=("NaiveBayes" -> nb_paramGrid)
  //---GBTClassifier---------------------------------------------------
  /**
    * GBT Classifier Object with label and feature column configured
    */
  val gbt = new GBTClassifier()
    .setLabelCol(label)
    .setFeaturesCol(featuresCol)

  var gbt_paramGrid = new ParamGridBuilder()
    .addGrid(gbt.maxDepth, ClassifiersManager.maxDepth)
    .addGrid(gbt.minInfoGain, ClassifiersManager.minInfoGain)
    .addGrid(gbt.minInstancesPerNode, ClassifiersManager.minInstancesPerNode )
    .build()

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

  var lda_paramGrid = new ParamGridBuilder()
    .addGrid(lda.scaledData ,Array(false) )
    .build()

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

  var qda_paramGrid = new ParamGridBuilder()
    .addGrid(qda.scaledData ,Array(true,false) )
    .build()

  ClassifierParamsMap +=("QDA" -> qda_paramGrid)
  ClassifiersMap += ( "QDA" -> qda)
  //---------------------------------------------------------------------


  val numTrees_dist = Poisson(60)(setSeed(seed)).sample(3)
  val maxDepth_dist = Poisson(12)(setSeed(seed)).sample(10)
  val minInfoGain_dist = Array(0.001, 0.01 , 0.1 ) //Gamma(1.5,0.1)(setSeed(seed)).sample(3)
  val impurity_dist    = Array("gini")
  val maxBins_dist =  Array(96)
  val minInstancesPerNode_dist= Array(1, 5, 10)
  val regParam_dist = Gamma(0.5,0.1 )(setSeed(seed)).sample(6)
  val elasticNetParam_dist = Gamma(0.5,0.1)(setSeed(seed)).sample(3)
  val maxIter_dist = Poisson(30)(setSeed(seed)).sample(4)
  val fitIntercept_dist =  Array(true,false)
  val standardization_dist = Array(true,false)
  val tol_dist = Array(1E-6)
  val smoothing_dist =  Array(1.0 , 0.9, 0.8, 0.7, 0.5 , 0.3, 0.1, 0.05)

  /**
    * generate parameters for certain classifier, the generated parameters are in the form
    * of Map( Index -> ParamMap) ex: Map( 1 -> (NumTrees, Depth, MaxBins, ...) )
    * @param classifier classifer
    * @return
    */
  def generateParametersFromList( classifier:String):ListBuffer[ParamMap] = {

    //var ClassifierParamsMapIndexed =  Map[ String,Map[Int,ParamMap] ]()
    var ClassifierParamsMapIndexed =  Map[ String, ListBuffer[ParamMap] ]()
    var ParamMapList = new ListBuffer[ParamMap]()
    classifier match {
      case "RandomForestClassifier" =>
        var rf_counter = 0
        var rf_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <- numTrees_dist )
        {
          for(j <- maxDepth_dist)
          {
            for (k <- maxBins_dist)
            {
              for ( l <- minInfoGain_dist)
              {
                for(m <- impurity_dist)
                {
                  for(n <- minInstancesPerNode_dist)
                    {
                      var pm = new ParamMap()
                      pm.put(rf.numTrees, i)
                      pm.put(rf.maxDepth, j)
                      pm.put(rf.maxBins, k)
                      pm.put(rf.minInfoGain, l)
                      pm.put(rf.impurity, m)
                      pm.put(rf.minInstancesPerNode, n)
                      rf_indexedParamMapArr += (rf_counter -> pm)
                      ParamMapList += pm
                      rf_counter = rf_counter + 1
                    }
                }
              }
            }
          }
        }
        //ClassifierParamsMapIndexed += ("RandomForestClassifier" -> rf_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("RandomForestClassifier" -> ParamMapList)

      case  "LogisticRegression" =>
        var lr_counter = 0
        var lr_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <- fitIntercept_dist )
        {
          for(j <- maxIter_dist)
          {
            for (k <- regParam_dist)
            {
              for ( l <- elasticNetParam_dist)
              {
                for(m <- standardization_dist) {
                  for( n <- tol_dist) {
                    var pm = new ParamMap()
                    pm.put(lr.fitIntercept, i)
                    pm.put(lr.maxIter, j)
                    pm.put(lr.regParam, k)
                    pm.put(lr.elasticNetParam, l)
                    pm.put(lr.standardization, m)
                    pm.put(lr.tol, n)
                    lr_indexedParamMapArr += (lr_counter -> pm)
                    ParamMapList += pm
                    lr_counter = lr_counter + 1
                  }
                }
              }
            }
          }
        }
        //ClassifierParamsMapIndexed += ("LogisticRegression" -> lr_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("LogisticRegression" -> ParamMapList)

      case "DecisionTreeClassifier" =>
        var dt_counter = 0
        var dt_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <-maxDepth_dist )
        {
          for(j <- maxBins_dist)
          {
            for (k <- minInfoGain_dist)
            {
              for ( l <- impurity_dist)
              {
                for(m <- minInstancesPerNode_dist) {
                  var pm = new ParamMap()
                  pm.put(dt.maxDepth, i)
                  pm.put(dt.maxBins, j)
                  pm.put(dt.minInfoGain, k)
                  pm.put(dt.impurity, l)
                  pm.put(dt.minInstancesPerNode, m)
                  dt_indexedParamMapArr += (dt_counter -> pm)
                  ParamMapList += pm
                  dt_counter = dt_counter + 1
                }
              }
            }
          }
        }
        //ClassifierParamsMapIndexed += ("DecisionTreeClassifier" -> dt_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("DecisionTreeClassifier" -> ParamMapList)

      case "MultilayerPerceptronClassifier" =>
        var mpr_counter = 0
        var mpr_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <-Array(
                        Array(nr_features,2,nr_classes),
                        Array(nr_features,3,nr_classes),
                        Array(nr_features,4,nr_classes),
                        Array(nr_features,5,nr_classes),
                        Array(nr_features,6,nr_classes),
                        Array(nr_features,7,nr_classes),
                        Array(nr_features,8,nr_classes),
                        Array(nr_features,9,nr_classes),
                        Array(nr_features,10,nr_classes)
                      )
            )
        {
          for(j <- maxIter_dist)
          {
            var pm = new ParamMap()
            pm.put(mpr.layers, i)
            pm.put(mpr.maxIter, j)
            mpr_indexedParamMapArr += (mpr_counter -> pm)
            ParamMapList += pm
            mpr_counter = mpr_counter + 1
          }
        }
        //ClassifierParamsMapIndexed += ("MultilayerPerceptronClassifier" -> mpr_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("MultilayerPerceptronClassifier" -> ParamMapList)

      case "LinearSVC" =>
        var lsvc_counter = 0
          var lsvc_indexedParamMapArr = Map[Int,ParamMap]()
          for( i <-maxIter_dist )
          {
            for(j <- regParam_dist)
            {
              var pm = new ParamMap()
              pm.put(lsvc.maxIter, i)
              pm.put(lsvc.regParam, j)
              lsvc_indexedParamMapArr += (lsvc_counter -> pm)
              ParamMapList += pm
              lsvc_counter = lsvc_counter + 1
            }
          }
          //ClassifierParamsMapIndexed += ("LinearSVC" -> lsvc_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("LinearSVC" -> ParamMapList)


      case "NaiveBayes" =>
        var nb_counter = 0
        var nb_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <- smoothing_dist )
        {
          var pm = new ParamMap()
          pm.put(nb.smoothing, i)
          nb_indexedParamMapArr += (nb_counter -> pm)
          ParamMapList += pm
          nb_counter = nb_counter + 1
        }
        //ClassifierParamsMapIndexed += ("NaiveBayes" -> nb_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("NaiveBayes" -> ParamMapList)

      case "GBTClassifier" =>
        var gbt_counter = 0
        var gbt_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <-maxDepth_dist )
        {
          for(j <- minInfoGain_dist)
          {
            for (k <- minInstancesPerNode_dist)
            {
              for( l <- maxBins_dist) {
                var pm = new ParamMap()
                pm.put(gbt.maxDepth, i)
                pm.put(gbt.minInfoGain, j)
                pm.put(gbt.minInstancesPerNode, k)
                pm.put(gbt.maxBins, l)
                gbt_indexedParamMapArr += (gbt_counter -> pm)
                ParamMapList += pm
                gbt_counter = gbt_counter + 1
              }
            }
          }
        }
        //ClassifierParamsMapIndexed += ("GBTClassifier" -> gbt_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("GBTClassifier" -> ParamMapList)

      case "LDA" =>
        var pm = new ParamMap()
        pm.put(lda.scaledData ,false)
        ParamMapList += pm
        //ClassifierParamsMapIndexed += ("LDA" -> Map(1 -> new ParamMap().put(lda.scaledData ,false)  ))
        ClassifierParamsMapIndexed += ("LDA" -> ParamMapList )

      case "QDA" =>
        var pm = new ParamMap()
        pm.put(lda.scaledData ,false)
        ParamMapList += pm
        //ClassifierParamsMapIndexed += ("LDA" -> Map(1 -> new ParamMap().put(lda.scaledData ,false)  ))
        ClassifierParamsMapIndexed += ("QDA" -> ParamMapList )

    }
    return ClassifierParamsMapIndexed(classifier)
  }
  def generateParametersFromDistribution( classifier:String):ListBuffer[ParamMap] = {

    //var ClassifierParamsMapIndexed =  Map[ String,Map[Int,ParamMap] ]()
    var ClassifierParamsMapIndexed =  Map[ String, ListBuffer[ParamMap] ]()
    var ParamMapList = new ListBuffer[ParamMap]()
    classifier match {
      case "RandomForestClassifier" =>
        var rf_counter = 0
        var rf_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <- ClassifiersManager.numTrees )
        {
          for(j <- ClassifiersManager.maxDepth)
          {
            for (k <- ClassifiersManager.maxBins)
            {
              for ( l <- ClassifiersManager.minInfoGain)
              {
                for(m <- ClassifiersManager.impurity)
                {
                  for(n <- ClassifiersManager.minInstancesPerNode)
                  {
                    var pm = new ParamMap()
                    pm.put(rf.numTrees, i)
                    pm.put(rf.maxDepth, j)
                    pm.put(rf.maxBins, k)
                    pm.put(rf.minInfoGain, l)
                    pm.put(rf.impurity, m)
                    pm.put(rf.minInstancesPerNode, n)
                    rf_indexedParamMapArr += (rf_counter -> pm)
                    ParamMapList += pm
                    rf_counter = rf_counter + 1
                  }
                }
              }
            }
          }
        }
        //ClassifierParamsMapIndexed += ("RandomForestClassifier" -> rf_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("RandomForestClassifier" -> ParamMapList)

      case  "LogisticRegression" =>
        var lr_counter = 0
        var lr_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <- ClassifiersManager.fitIntercept )
        {
          for(j <- ClassifiersManager.maxIter)
          {
            for (k <- ClassifiersManager.regParam)
            {
              for ( l <- ClassifiersManager.elasticNetParam)
              {
                for(m <- ClassifiersManager.standardization) {
                  for( n <- ClassifiersManager.tol) {
                    var pm = new ParamMap()
                    pm.put(lr.fitIntercept, i)
                    pm.put(lr.maxIter, j)
                    pm.put(lr.regParam, k)
                    pm.put(lr.elasticNetParam, l)
                    pm.put(lr.standardization, m)
                    pm.put(lr.tol, n)
                    lr_indexedParamMapArr += (lr_counter -> pm)
                    ParamMapList += pm
                    lr_counter = lr_counter + 1
                  }
                }
              }
            }
          }
        }
        //ClassifierParamsMapIndexed += ("LogisticRegression" -> lr_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("LogisticRegression" -> ParamMapList)

      case "DecisionTreeClassifier" =>
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
                for(m <- ClassifiersManager.minInstancesPerNode) {
                  var pm = new ParamMap()
                  pm.put(dt.maxDepth, i)
                  pm.put(dt.maxBins, j)
                  pm.put(dt.minInfoGain, k)
                  pm.put(dt.impurity, l)
                  pm.put(dt.minInstancesPerNode, m)
                  dt_indexedParamMapArr += (dt_counter -> pm)
                  ParamMapList += pm
                  dt_counter = dt_counter + 1
                }
              }
            }
          }
        }
        //ClassifierParamsMapIndexed += ("DecisionTreeClassifier" -> dt_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("DecisionTreeClassifier" -> ParamMapList)

      case "MultilayerPerceptronClassifier" =>
        var mpr_counter = 0
        var mpr_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <-Array(
          Array(nr_features,2,nr_classes),
          Array(nr_features,3,nr_classes),
          Array(nr_features,4,nr_classes),
          Array(nr_features,5,nr_classes),
          Array(nr_features,6,nr_classes),
          Array(nr_features,7,nr_classes),
          Array(nr_features,8,nr_classes),
          Array(nr_features,9,nr_classes),
          Array(nr_features,10,nr_classes)
        )
        )
        {
          for(j <- ClassifiersManager.maxIter)
          {
            var pm = new ParamMap()
            pm.put(mpr.layers, i)
            pm.put(mpr.maxIter, j)
            mpr_indexedParamMapArr += (mpr_counter -> pm)
            ParamMapList += pm
            mpr_counter = mpr_counter + 1
          }
        }
        //ClassifierParamsMapIndexed += ("MultilayerPerceptronClassifier" -> mpr_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("MultilayerPerceptronClassifier" -> ParamMapList)

      case "LinearSVC" =>
        var lsvc_counter = 0
        var lsvc_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <-ClassifiersManager.maxIter )
        {
          for(j <- ClassifiersManager.regParam)
          {
            var pm = new ParamMap()
            pm.put(lsvc.maxIter, i)
            pm.put(lsvc.regParam, j)
            lsvc_indexedParamMapArr += (lsvc_counter -> pm)
            ParamMapList += pm
            lsvc_counter = lsvc_counter + 1
          }
        }
        //ClassifierParamsMapIndexed += ("LinearSVC" -> lsvc_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("LinearSVC" -> ParamMapList)


      case "NaiveBayes" =>
        var nb_counter = 0
        var nb_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <- ClassifiersManager.smoothing )
        {
          var pm = new ParamMap()
          pm.put(nb.smoothing, i)
          nb_indexedParamMapArr += (nb_counter -> pm)
          ParamMapList += pm
          nb_counter = nb_counter + 1
        }
        //ClassifierParamsMapIndexed += ("NaiveBayes" -> nb_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("NaiveBayes" -> ParamMapList)

      case "GBTClassifier" =>
        var gbt_counter = 0
        var gbt_indexedParamMapArr = Map[Int,ParamMap]()
        for( i <-ClassifiersManager.maxDepth )
        {
          for(j <- ClassifiersManager.minInfoGain)
          {
            for (k <- ClassifiersManager.minInstancesPerNode)
            {
              for( l <- ClassifiersManager.maxBins) {
                var pm = new ParamMap()
                pm.put(gbt.maxDepth, i)
                pm.put(gbt.minInfoGain, j)
                pm.put(gbt.minInstancesPerNode, k)
                pm.put(gbt.maxBins, l)
                gbt_indexedParamMapArr += (gbt_counter -> pm)
                ParamMapList += pm
                gbt_counter = gbt_counter + 1
              }
            }
          }
        }
        //ClassifierParamsMapIndexed += ("GBTClassifier" -> gbt_indexedParamMapArr)
        ClassifierParamsMapIndexed += ("GBTClassifier" -> ParamMapList)

      case "LDA" =>
        var pm = new ParamMap()
        pm.put(lda.scaledData ,false)
        ParamMapList += pm
        //ClassifierParamsMapIndexed += ("LDA" -> Map(1 -> new ParamMap().put(lda.scaledData ,false)  ))
        ClassifierParamsMapIndexed += ("LDA" -> ParamMapList )

      case "QDA" =>
        var pm = new ParamMap()
        pm.put(lda.scaledData ,false)
        ParamMapList += pm
        //ClassifierParamsMapIndexed += ("LDA" -> Map(1 -> new ParamMap().put(lda.scaledData ,false)  ))
        ClassifierParamsMapIndexed += ("QDA" -> ParamMapList )

    }
    return ClassifierParamsMapIndexed(classifier)
  }


  /**
    * select n parameters randomly from list of indexed parameters
    * @param classifier classifier
    * @param n number of config (parameters) to be selected randomly
    * @param s number to be added to the seed
    */
  def getNRandomParameters(classifier:String,n:Int,paramSource:Int = 0 , s:Int = 1): Map[Int,ParamMap] =   {

    // select random n hyper parameters configuration
    var IndexedRandomParamMapList: ListBuffer[ParamMap]  = null
    if(paramSource == 0)
      IndexedRandomParamMapList =    generateParametersFromDistribution(classifier)
    else
      IndexedRandomParamMapList = generateParametersFromList(classifier)

    val end   = IndexedRandomParamMapList.size

    val r = new scala.util.Random(seed + s)
    var lstRandomParam = ListBuffer[Int]()
    var iter = 0

    var SelectedParamsMapIndexed =  Map[Int,ParamMap]()
    var SelectedParamsMapList    =  ListBuffer[ParamMap]()

    //in case of all the parameters less than the needed parametr by hyperband first sh
    if( end <= n) {
      SelectedParamsMapList  = IndexedRandomParamMapList
      for(elm <- IndexedRandomParamMapList) {
        SelectedParamsMapIndexed += (iter -> elm)
        iter = iter + 1
      }
    }
    else
    {
      while (lstRandomParam.size < n) {
        var num = r.nextInt(end)
        if( !lstRandomParam.contains(num))
          lstRandomParam += num
      }

      for (x <- lstRandomParam) {
        SelectedParamsMapList+= IndexedRandomParamMapList(x)
        SelectedParamsMapIndexed += (iter -> IndexedRandomParamMapList(x))
        iter = iter + 1
       }
     }


    /*println("====================Selected Parameters============================")
    var s12 = ""
    for ( ps <- SelectedParamsMapIndexed) {
      for (p <- ps._2.toSeq) {
        s12 = p.param.name + ":" + p.value + "," + s12

      }
      println(s12)
      s12 = ""
    }
      println("=============================================================")*/

    return SelectedParamsMapIndexed
  }

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





  def SortedSelectedClassifiers( lst:Array[Double]) : List[Int] =  {
    var rslt= List[Int]()
    for( i <- OrderList ) {
        if ( lst.contains(i))
          rslt = i :: rslt
      }

  return rslt.reverse
  }



}
