//package org.apache.spark.ml.tuning
package org.dsmartml

import org.apache.spark.ml.Model
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.Hyperband
import org.apache.spark.sql.{DataFrame, SparkSession}

/**
  * Class Name: Logger
  * Description: this calss responable for logging information to text file
  *            :  it log (Time Log, Exception and Knowledgebase output)
  * @constructor Create a MetadataManager object allow us to manage (extract) our dataset metadata (
  * @param spark the used spark session
  * @param logger Object of the logger class (used to log time, exception and result)
  * @param TargetCol the dataset label column name (default = y)
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/3/2019
  */
class ModelSelector (spark:SparkSession, logger:Logger, featureCol:String = "features" , TargetCol:String = "y" ,
                     eta:Int = 3, maxResourcePercentage:Int = 100, Parallelism:Int = 2,
                     seed:Long=1234 , ConvertToVecAssembly:Boolean = true,
                     ScaleData:Boolean = true  , TryNClassifier:Int = 2 , HP_MaxTime:Long = 100000) {

  /**
    * this function get the best model and its hyper-parametrers and accuracy
    * @param df Input dataset
    * @return best algorithm name, Model, best hyper parameters and accuracy
    */
  def getBestModel(df:DataFrame ): (String, ( Model[_] , ParamMap , Double)) = {


    // Creare Knowlegdebase Manager
    var kbmgr = new KBManager(spark, logger, TargetCol)

    // Get Best Algorithms based on the Knowledgebase
    val selectedClassifiers = kbmgr.PredictBestClassifiers(df)

    // Create Classifiers Manager
    val ClassifierMgr = new ClassifiersManager(spark, kbmgr._metadata.nr_features, kbmgr._metadata.nr_classes)

    //prepare dataset by converting to Vector Assembly & Scale it (if needed)
    var mydataset = df
    if (ConvertToVecAssembly)
      mydataset = DataLoader.convertDFtoVecAssembly(df, featureCol, TargetCol)
    if (ScaleData)
      mydataset = DataLoader.ScaleDF(mydataset, featureCol, TargetCol)

    mydataset.persist()
    //split data
    val Array(trainingData_MinMaxScaled, testData_MinMaxScaled) = mydataset.randomSplit(Array(0.8, 0.2), seed)


    // return best Model with its parameters and accuracy
    //return Map(selectedModelMap.toSeq.sortWith(_._2._3 > _._2._3):_*).head //._2._1
    //return (modelname, ( null , pm , acc))
    //if (selectedModelMap.head != null)
    //  return selectedModelMap.head
    //else return null
    var x: (String, (Model[_], ParamMap, Double)) = null
    for (time <- Array(50, 100, 200, 300, 400)) {
      try {
         x = HyperParametersOpt(mydataset, selectedClassifiers, kbmgr, ClassifierMgr, time)
      }
      catch {
        case ex: Exception => println(ex.getMessage)
        //                         logger.logOutput("Exception: " + ex.getMessage)
         logger.close()
      }

    }
    return x
  }

  def HyperParametersOpt(mydataset:DataFrame , selectedClassifiers:List[Int] , kbmgr:KBManager ,ClassifierMgr:ClassifiersManager, t:Int ): (String, ( Model[_] , ParamMap , Double)) = {

    val StartTime = new java.util.Date().getTime
    // output
    var modelname: String = ""
    var bestmodel :Model[_] = null
    var pm:ParamMap = null
    var acc:Double = 0.0

    //split data
    val Array(trainingData_MinMaxScaled, testData_MinMaxScaled) = mydataset.randomSplit(Array(0.8, 0.2) , seed)

    //Cerate Hyperband
    val hb = new Hyperband()

    //Create Map to save each model and its parameters and its accuracy
    var selectedModelMap = Map[String, ( Model[_] , ParamMap , Double)]()

    // for each good Algorithm (based on KB) use Hyperband
    println("3 - Hyperband for Selected Algorithms")


    for( i <- selectedClassifiers) {// selectedClassifiers_ordered.sorted[Int].take(TryNClassifier) ) {
      //var classifier_order:Int = ClassifiersManager.classifiersOrderMap( ClassifiersManager.classifiersLsit(i.toInt))

      if ((kbmgr._metadata.nr_classes == 2 && Array(4, 6).contains(i))
        ||
        (kbmgr._metadata.nr_classes >= 2 && Array(0, 1, 2, 3).contains(i))
        ||
        (kbmgr._metadata.hasNegativeFeatures == false && i == 5)
      ) {
        //if( i != 0  && i != 3){
        try {

          if (!hb.IsTimeOut()) {
            val starttime1 = new java.util.Date().getTime
            var classifier = ClassifiersManager.classifiersLsit(i) // ClassifiersManager.classifiersOrderLsit(i.toInt)
            hb.setEstimator(ClassifierMgr.ClassifiersMap(classifier))
            hb.setEvaluator(ClassifierMgr.evaluator)
            hb.setEstimatorParamMaps(ClassifierMgr.ClassifierParamsMap(classifier))
            hb.setEta(eta)
            hb.setmaxResource(maxResourcePercentage)
            hb.setLogFilePath("/home/sshuser/result.txt")
            hb.setLogToFile(false)
            hb.setCollectSubModels(false)
            hb.setParallelism(Parallelism)
            hb.setClassifierName(classifier)
            hb.setmaxTime(t)
            hb.filelog = logger
            hb.ClassifierParamsMapIndexed = ClassifierMgr.ClassifierParamsMapIndexed(classifier)
            hb.ClassifiersMgr = ClassifierMgr
            println("   -- Start Hyperband for " + classifier)
            val model = hb.fit(mydataset)//trainingData_MinMaxScaled)

            //val predictions = model.bestModel.transform(testData_MinMaxScaled)
            //var accuracy = ClassifierMgr.evaluator.evaluate(predictions)


            var accuracy = hb.bestmetric
            //val accuracy = model.validationMetrics(0)
            i match {

              //Random Forest
              case 0 =>
                if (acc < accuracy) {
                  selectedModelMap = Map(classifier -> (model.bestModel.asInstanceOf[RandomForestClassificationModel], model.bestModel.extractParamMap(), accuracy))
                  acc = accuracy
                  modelname = classifier
                  bestmodel = model.bestModel.asInstanceOf[RandomForestClassificationModel]
                  pm = model.bestModel.extractParamMap()

                }

              //Logestic Regression
              case 1 =>
                if (acc < accuracy) {
                  selectedModelMap = Map(classifier -> (model.bestModel.asInstanceOf[LogisticRegressionModel], model.bestModel.extractParamMap(), accuracy))
                  acc = accuracy
                  modelname = classifier
                  bestmodel = model.bestModel.asInstanceOf[LogisticRegressionModel]
                  pm = model.bestModel.extractParamMap()

                }

              //Decision Tree
              case 2 =>
                if (acc < accuracy) {
                  selectedModelMap = Map(classifier -> (model.bestModel.asInstanceOf[DecisionTreeClassificationModel], model.bestModel.extractParamMap(), accuracy))
                  acc = accuracy
                  modelname = classifier
                  bestmodel = model.bestModel.asInstanceOf[DecisionTreeClassificationModel]
                  pm = model.bestModel.extractParamMap()

                }

              //MultiLayer perceptron
              case 3 =>
                if (acc < accuracy) {
                  selectedModelMap = Map(classifier -> (model.bestModel.asInstanceOf[MultilayerPerceptronClassificationModel], model.bestModel.extractParamMap(), accuracy))
                  acc = accuracy
                  modelname = classifier
                  bestmodel = model.bestModel.asInstanceOf[MultilayerPerceptronClassificationModel]
                  pm = model.bestModel.extractParamMap()

                }

              //Linear SVC
              case 4 =>
                if (acc < accuracy) {
                  selectedModelMap = Map(classifier -> (model.bestModel.asInstanceOf[LinearSVCModel], model.bestModel.extractParamMap(), accuracy))
                  acc = accuracy
                  modelname = classifier
                  bestmodel = model.bestModel.asInstanceOf[LinearSVCModel]
                  pm = model.bestModel.extractParamMap()

                }

              //Naive bayes
              case 5 =>
                if (acc < accuracy) {
                  selectedModelMap = Map(classifier -> (model.bestModel.asInstanceOf[NaiveBayesModel], model.bestModel.extractParamMap(), accuracy))
                  acc = accuracy
                  modelname = classifier
                  bestmodel = model.bestModel.asInstanceOf[NaiveBayesModel]
                  pm = model.bestModel.extractParamMap()

                }

              //GBT
              case 6 =>
                if (acc < accuracy) {
                  selectedModelMap = Map(classifier -> (model.bestModel.asInstanceOf[GBTClassificationModel], model.bestModel.extractParamMap(), accuracy))
                  acc = accuracy
                  modelname = classifier
                  bestmodel = model.bestModel.asInstanceOf[GBTClassificationModel]
                  pm = model.bestModel.extractParamMap()

                }

            }


            //println("Algorithm:" + classifier)
            //println("Accuracy:" + accuracy)
            //println("========================================================")
            val Endtime1 = new java.util.Date().getTime
            val TotalTime1 = Endtime1 - starttime1
            println("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ")  Accuracy: " + accuracy)
          }
        } catch {
          case ex: Exception => println("-- Hyperband for algoritm: Exception " + ex.getMessage)
        }
        // }
      }

      // For LDA
      if (i == 7 && ! hb.IsTimeOut()) {
        try {
          val starttime1 = new java.util.Date().getTime
          var classifier = ClassifiersManager.classifiersLsit(i.toInt)
          // evaluator
          val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol(TargetCol)
            .setPredictionCol("prediction")
            .setMetricName("accuracy")

          val lda = new org.apache.spark.ml.classification.LDA()
          lda.sc = spark.sparkContext
          lda.setLabelCol(TargetCol)
          lda.setFeaturesCol(featureCol)
          lda.setScaledData(false)
          lda.setPredictionCol("prediction")
          lda.setProbabilityCol("Probability")
          lda.setRawPredictionCol("RawPrediction")
          //lda.sc = spark.sparkContext
          val model_lda = lda.fit(trainingData_MinMaxScaled)
          val predictions_lda = model_lda.transform(testData_MinMaxScaled)
          val accuracy_lda = evaluator.evaluate(predictions_lda)

          if (acc < accuracy_lda) {
            selectedModelMap = Map(classifier -> (model_lda, null, accuracy_lda))
            acc = accuracy_lda
            modelname = classifier
            bestmodel = null //model_lda.asInstanceOf[org.apache.spark.ml.classification.LDA]
            pm = null

          }

          val Endtime1 = new java.util.Date().getTime
          val TotalTime1 = Endtime1 - starttime1
          println("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ") Accuracy: " + accuracy_lda)
        } catch {
          case ex: Exception => println("-- Hyperband for algoritm: LDA Exception " + ex.getMessage)
        }
      }

      //For QDA
      if (i == 8 && ! hb.IsTimeOut()) {
        try {
          val starttime1 = new java.util.Date().getTime
          var classifier = ClassifiersManager.classifiersLsit(i.toInt)
          // evaluator
          val evaluator = new MulticlassClassificationEvaluator()
            .setLabelCol(TargetCol)
            .setPredictionCol("prediction")
            .setMetricName("accuracy")

          val qda = new org.apache.spark.ml.classification.QDA(spark.sparkContext)
          qda.setLabelCol(TargetCol)
          qda.setFeaturesCol(featureCol)
          qda.setScaledData(false)
          qda.setPredictionCol("prediction")
          qda.setProbabilityCol("Probability")
          qda.setRawPredictionCol("RawPrediction")
          val model_qda = qda.fit(trainingData_MinMaxScaled)
          val predictions_qda = model_qda.transform(testData_MinMaxScaled)
          val accuracy_qda = evaluator.evaluate(predictions_qda)


          if (acc < accuracy_qda) {
            selectedModelMap = Map(classifier -> (model_qda, null, accuracy_qda))
            acc = accuracy_qda
            modelname = classifier
            bestmodel = null //model_qda
            pm = null

          }

          val Endtime1 = new java.util.Date().getTime
          val TotalTime1 = Endtime1 - starttime1
          println("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ")  Accuracy: " + accuracy_qda)
        } catch {
          case ex: Exception => println("-- Hyperband for algoritm: QDA Exception " + ex.getMessage)

        }
      }

    }


    // return best Model with its parameters and accuracy
    //return Map(selectedModelMap.toSeq.sortWith(_._2._3 > _._2._3):_*).head //._2._1
    //return (modelname, ( null , pm , acc))
    if (selectedModelMap.head != null) {
      val Endtime = new java.util.Date().getTime
      val TotalTime1 = Endtime - StartTime
      println("  =>Result:")
      println("  |--Best Algorithm:" + selectedModelMap.head._1)
      println("  |--Accuracy:" + selectedModelMap.head._2._3)
      println("  |--Total Time:" + (TotalTime1 / 1000.0).toString)
      var plist = ""
        for ( d <- selectedModelMap.head._2._2.toSeq.toList)
        {
          if(plist != "")
            plist = d.param.name + ":" + d.value + "," + plist
          else
            plist = d.param.name + ":" + d.value
        }

      logger.logOutput("Max Time:" + t + ", Total Time:" +  (TotalTime1 / 1000.0).toString + ", eta:" + eta+  ", Accuracy:" + selectedModelMap.head._2._3 + ", Parameters:(" + plist  + ") \n"  )
      return selectedModelMap.head
    }
    else return null
  }


}
