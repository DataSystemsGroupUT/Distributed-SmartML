//package org.apache.spark.ml.tuning
package org.dsmartml

import java.text.DecimalFormat
import java.util.Date
import org.apache.spark.ml.Model
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.sql.{DataFrame, SparkSession}
import org.dsmartml.knowledgeBase.KBManager

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
class ModelSelector (spark:SparkSession,
                     logpath:String,
                     featureCol:String = "features" ,
                     TargetCol:String = "y" ,
                     eta:Int = 5,
                     MaxRandomSearchParam:Int = 20,
                     maxResourcePercentage:Int = 100,
                     Parallelism:Int = 3,
                     seed:Int=1234 ,
                     ConvertToVecAssembly:Boolean = true,
                     ScaleData:Boolean = true  ,
                     TryNClassifier:Int = 6 ,
                     HPOptimizer:Int = 2,
                     skip_SH:Int = 0,
                     SplitbyClass:Boolean = false,
                     basicDataPercentage:Double = 0,
                     var HP_MaxTime:Long = 100000) {





  var StartingTime: Date = new Date()
  val fm2d = new DecimalFormat("###.##")
  val fm4d = new DecimalFormat("###.####")
  var logger = new Logger(logpath)

  /**
    * check if timeout or not
    * @return
    */
  def IsTimeOut(): Boolean = {
    if (getRemainingTime() == 0)
      return true
    else
      return false
  }

  /**
    * calculate remaining time
    * @return
    */
  def getRemainingTime(): Long = {
    var rem: Long = (HP_MaxTime * 1000) - (new Date().getTime - StartingTime.getTime())
    if (rem < 0)
      rem = 0
    return rem
  }


  /**
    * this function get the best model and its hyper-parametrers and accuracy
    * @param df Input dataset
    * @return best algorithm name, Model, best hyper parameters and accuracy
    */
  def getBestModel(df:DataFrame ): (String, ( Model[_] , ParamMap , Double)) = {

    logger.printHeader(HPOptimizer, HP_MaxTime, Parallelism, eta, maxResourcePercentage, MaxRandomSearchParam,seed)
    logger.logHeader(HPOptimizer, HP_MaxTime, Parallelism, eta, maxResourcePercentage, MaxRandomSearchParam,seed)

    // Creare Knowlegdebase Manager
    var kbmgr = new KBManager(spark, logger, TargetCol)

    // Get Best Algorithms based on the Knowledgebase
    val selectedClassifiers = kbmgr.PredictBestClassifiers(df)

    // Create Classifiers Manager
    val ClassifierMgr = new ClassifiersManager(spark, kbmgr._metadata.nr_features, kbmgr._metadata.nr_classes, label = TargetCol, seed = seed.toInt)

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


    /*
    for (time <- Array(100, 200, 300, 400)) {
      try {
         x = HyperParametersOpt(mydataset, selectedClassifiers, kbmgr, ClassifierMgr, time)
      }
      catch {
        case ex: Exception => println(ex.getMessage)
        //                         logger.logOutput("Exception: " + ex.getMessage)
         logger.close()
      }

    }
  */
    //for (time <-  Array(100,200,300)) { //Array(100,300,600,1800)) {
      try {

        //HP_MaxTime = time
        //println("     === Time:" + HP_MaxTime)
        StartingTime = new Date()
        x = HyperParametersOpt(mydataset, selectedClassifiers, kbmgr, ClassifierMgr, HP_MaxTime.toInt)//HP_MaxTime.toInt)
      }
    catch
    {
      case ex: Exception => println(ex.getMessage)
        //                         logger.logOutput("Exception: " + ex.getMessage)
        //logger.close()
    }
      //logger.close()
  //}
    logger.close()
    return x
  }


  /**
    *
    * @param mydataset
    * @param ClassifierMgr
    * @param classifier
    * @param StartingTime
    * @return
    */
  def getAlgorithmBestModel(
                             mydataset:DataFrame,
                             ClassifierMgr:ClassifiersManager,
                             classifier:String,
                             StartingTime:Date
                           ) :(ParamMap, Double, Model[_]) = {

    // hyperband
    if (HPOptimizer == 1) {
      // for each good Algorithm (based on KB) use Hyperband
      //println("3 - Hyperband")
      val hb = new org.apache.spark.ml.tuning.Hyperband()
      hb.StartingTime = StartingTime
      if (!IsTimeOut()) {
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
        hb.setmaxTime(HP_MaxTime)
        hb.setskipSH(skip_SH)
        hb.setSplitbyClass(SplitbyClass)
        hb.setbasicDataPercentage(basicDataPercentage)
        hb.filelog = logger
        //hb.ClassifierParamsMapIndexed = ClassifierMgr.ClassifierParamsMapIndexed(classifier)
        hb.ClassifiersMgr = ClassifierMgr
        println("   -- Start Hyperband for " + classifier)
        val model = hb.fit(mydataset)
        if (classifier == "RandomForestClassifier")
          return (hb.bestParam, hb.bestmetric, hb.bestModel.asInstanceOf[RandomForestClassificationModel])
        else if (classifier == "LogisticRegression")
          return (hb.bestParam, hb.bestmetric, hb.bestModel.asInstanceOf[LogisticRegressionModel])
        else if (classifier== "DecisionTreeClassifier")
          return (hb.bestParam, hb.bestmetric, hb.bestModel.asInstanceOf[DecisionTreeClassificationModel])
        else if (classifier == "MultilayerPerceptronClassifier")
          return (hb.bestParam, hb.bestmetric, hb.bestModel.asInstanceOf[MultilayerPerceptronClassificationModel])
        else if (classifier == "LinearSVC")
          return (hb.bestParam, hb.bestmetric, hb.bestModel.asInstanceOf[LinearSVCModel])
        else if (classifier == "NaiveBayes")
          return (hb.bestParam, hb.bestmetric, hb.bestModel.asInstanceOf[NaiveBayesModel])
        else if (classifier == "GBTClassifier")
          return (hb.bestParam, hb.bestmetric, hb.bestModel.asInstanceOf[GBTClassificationModel])
        else if (classifier == "LDA")
          return (hb.bestParam, hb.bestmetric, hb.bestModel.asInstanceOf[LDAModel])
        else
          return (hb.bestParam, hb.bestmetric, hb.bestModel.asInstanceOf[QDAModel])

      }
      else
        return null
    }
      //random Search
      else if (HPOptimizer == 2) {
        //println("3 - Random Search")
        val rs = new org.apache.spark.ml.tuning.RandomSearch()
        rs.StartingTime = StartingTime
        if (!IsTimeOut()) {
          //println(" ------>Remaining time" + rs.getRemainingTime())
          rs.setmaxResource(maxResourcePercentage)
          rs.setLogFilePath("/home/sshuser/result.txt")
          rs.setLogToFile(false)
          rs.setCollectSubModels(false)
          rs.setParallelism(Parallelism)
          rs.setClassifierName(classifier)
          rs.setmaxTime(HP_MaxTime)
          rs.setTargetColumn(TargetCol)
          rs.filelog = logger
          rs.spark = spark
          rs.ClassifiersMgr = ClassifierMgr
          rs.setParamNumber(MaxRandomSearchParam)
          println("   -- Start Random Search for " + classifier)
          val model = rs.fit(mydataset)
          if (classifier == "RandomForestClassifier")
            return (rs.bestParam, rs.bestmetric, rs.bestModel.asInstanceOf[RandomForestClassificationModel])
          else if (classifier == "LogisticRegression")
            return (rs.bestParam, rs.bestmetric, rs.bestModel.asInstanceOf[LogisticRegressionModel])
          else if (classifier== "DecisionTreeClassifier")
            return (rs.bestParam, rs.bestmetric, rs.bestModel.asInstanceOf[DecisionTreeClassificationModel])
          else if (classifier == "MultilayerPerceptronClassifier")
            return (rs.bestParam, rs.bestmetric, rs.bestModel.asInstanceOf[MultilayerPerceptronClassificationModel])
          else if (classifier == "LinearSVC")
            return (rs.bestParam, rs.bestmetric, rs.bestModel.asInstanceOf[LinearSVCModel])
          else if (classifier == "NaiveBayes")
            return (rs.bestParam, rs.bestmetric, rs.bestModel.asInstanceOf[NaiveBayesModel])
          else if (classifier == "GBTClassifier")
            return (rs.bestParam, rs.bestmetric, rs.bestModel.asInstanceOf[GBTClassificationModel])
          else if (classifier == "LDA")
            return (rs.bestParam, rs.bestmetric, rs.bestModel.asInstanceOf[LDAModel])
          else
            return (rs.bestParam, rs.bestmetric, rs.bestModel.asInstanceOf[QDAModel])
        }
        else
          return null
      }
      else
        return null
  }

  /**
    * Hyperparameter optimization
    * @param mydataset
    * @param selectedClassifiers
    * @param kbmgr
    * @param ClassifierMgr
    * @param t
    * @return
    */
  def HyperParametersOpt(mydataset:DataFrame , selectedClassifiers:List[Int] , kbmgr:KBManager ,ClassifierMgr:ClassifiersManager, t:Int ): (String, ( Model[_] , ParamMap , Double)) = {

    // starting time for optimization
    val StartTime = new java.util.Date().getTime


    //val Start = new java.util.Date()
    // output
    var modelname: String = ""
    var bestmodel :Model[_] = null
    var pm:ParamMap = null
    var acc:Double = 0.0

    //split data
    val Array(trainingData_MinMaxScaled, testData_MinMaxScaled) = mydataset.randomSplit(Array(0.8, 0.2) , seed)

    //Create Map to save each model and its parameters and its accuracy
    var selectedModelMap = Map[String, ( Model[_] , ParamMap , Double)]()

    if(HPOptimizer == 1) {
      println("3 - Hyper-parameters Optimization using Hyperband with Time budget (" + HP_MaxTime +") Second.")
      logger.logOutput("3 - Hyper-parameters Optimization using Hyperband with Time budget (" + HP_MaxTime +") Second.\n")
    }
    else if(HPOptimizer == 2) {
      println("3 - Hyper-parameters Optimization using Random Search with Time budget (" + HP_MaxTime +") Second.")
      logger.logOutput("3 - Hyper-parameters Optimization using Random Search with Time budget (" + HP_MaxTime +") Second.\n")
    }

    //StartingTime= new Date()
    for( i <- selectedClassifiers) {

      if (  ((kbmgr._metadata.nr_classes == 2 && Array(4, 6).contains(i))
                ||
            (kbmgr._metadata.nr_classes >= 2 && Array(0, 1, 2, 3).contains(i))
               ||
            (kbmgr._metadata.hasNegativeFeatures == false && i == 5))
              &&
             !IsTimeOut()
      ) {

        try {
          val starttime1 = new java.util.Date().getTime
          var classifier = ClassifiersManager.classifiersLsit(i) // ClassifiersManager.classifiersOrderLsit(i.toInt)
          var result = getAlgorithmBestModel(trainingData_MinMaxScaled, ClassifierMgr, classifier, StartingTime)

          if(result != null  && result._1 != null && result._2 != null && result._3 != null)
          {
            var accuracy = result._2
            var selectedModel = result._3
            var selectedParamMap = result._1


            i match {
            //Random Forest
            case 0 =>
              if (acc < accuracy) {
                selectedModelMap = Map(classifier -> (selectedModel.asInstanceOf[RandomForestClassificationModel], selectedParamMap, accuracy))
                acc = accuracy
                modelname = classifier
                bestmodel = selectedModel.asInstanceOf[RandomForestClassificationModel]
                pm = selectedParamMap

              }

            //Logestic Regression
            case 1 =>
              if (acc < accuracy) {
                selectedModelMap = Map(classifier -> (selectedModel.asInstanceOf[LogisticRegressionModel], selectedParamMap, accuracy))
                acc = accuracy
                modelname = classifier
                bestmodel = selectedModel.asInstanceOf[LogisticRegressionModel]
                pm = selectedParamMap

              }

            //Decision Tree
            case 2 =>
              if (acc < accuracy) {
                selectedModelMap = Map(classifier -> (selectedModel.asInstanceOf[DecisionTreeClassificationModel], selectedParamMap, accuracy))
                acc = accuracy
                modelname = classifier
                bestmodel = selectedModel.asInstanceOf[DecisionTreeClassificationModel]
                pm = selectedParamMap

              }

            //MultiLayer perceptron
            case 3 =>
              if (acc < accuracy) {
                selectedModelMap = Map(classifier -> (selectedModel.asInstanceOf[MultilayerPerceptronClassificationModel], selectedParamMap, accuracy))
                acc = accuracy
                modelname = classifier
                bestmodel = selectedModel.asInstanceOf[MultilayerPerceptronClassificationModel]
                pm = selectedParamMap

              }

            //Linear SVC
            case 4 =>
              if (acc < accuracy) {
                selectedModelMap = Map(classifier -> (selectedModel.asInstanceOf[LinearSVCModel], selectedParamMap, accuracy))
                acc = accuracy
                modelname = classifier
                bestmodel = selectedModel.asInstanceOf[LinearSVCModel]
                pm = selectedParamMap

              }

            //Naive bayes
            case 5 =>
              if (acc < accuracy) {
                selectedModelMap = Map(classifier -> (selectedModel.asInstanceOf[NaiveBayesModel], selectedParamMap, accuracy))
                acc = accuracy
                modelname = classifier
                bestmodel = selectedModel.asInstanceOf[NaiveBayesModel]
                pm = selectedParamMap

              }

            //GBT
            case 6 =>
              if (acc < accuracy) {
                selectedModelMap = Map(classifier -> (selectedModel.asInstanceOf[GBTClassificationModel], selectedParamMap, accuracy))
                acc = accuracy
                modelname = classifier
                bestmodel = selectedModel.asInstanceOf[GBTClassificationModel]
                pm = selectedParamMap
              }
            }

          val Endtime1 = new java.util.Date().getTime
          val TotalTime1 = Endtime1 - starttime1
          println("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ") Accuracy: " + fm4d.format(100 * accuracy) + "%")
          logger.logOutput("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ") Accuracy: " + fm4d.format(100 * accuracy) + "%\n")
          logger.logLastResult("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ") Accuracy: " + fm4d.format(100 * accuracy) + "Params "+ selectedParamMap.toString()  + "%\n")

          }
          else
            {
              val Endtime1 = new java.util.Date().getTime
              val TotalTime1 = Endtime1 - starttime1
              println("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ") Accuracy: 0.00%")
              logger.logOutput("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ") Accuracy: 0.00%\n")

            }
        } catch {
          case ex: Exception => println("-- Hyper-Param  Optimization Exception :" + ex.getMessage)
        }
      }

      // For LDA
      if (i == 7 && !IsTimeOut()) {
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
          println("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ") Accuracy: " + fm4d.format(100 * accuracy_lda) + "%")
          logger.logOutput("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ") Accuracy: " + fm4d.format(100 * accuracy_lda) + "%\n")
        } catch {
          case ex: Exception => println("-- Hyperband for algoritm: LDA Exception " + ex.getMessage)
        }
      }

      //For QDA
      if (i == 8 && !IsTimeOut()) {
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
          println("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ")  Accuracy: " + fm4d.format(100 * accuracy_qda) + "%" )
          logger.logOutput("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ")  Accuracy: " + fm4d.format(100 * accuracy_qda) + "%\n")
        } catch {
          case ex: Exception => println("-- Hyperband for algoritm: QDA Exception " + ex.getMessage)

        }
      }

    }

    // return best Model with its parameters and accuracy
    if (selectedModelMap.size > 0) {
      var plist = ""
      for ( d <- selectedModelMap.head._2._2.toSeq.toList)
      {
        if(plist != "")
          plist = d.param.name + ":" + d.value + "," + plist
        else
          plist = d.param.name + ":" + d.value
      }

      val Endtime = new java.util.Date().getTime
      val TotalTime1 = Endtime - StartingTime.getTime
      logger.printLine()
      logger.logLine()
      println("  =>Result:")
      println("  |--Best Algorithm:" + selectedModelMap.head._1)
      println("  |--Accuracy:" + fm4d.format( 100 * selectedModelMap.head._2._3) + "%")
      println("  |--Total Time:" + (TotalTime1 / 1000.0).toString)
      println("  |--Best Parameters:")
      println("          " + plist)

      logger.logOutput("  =>Result:\n")
      logger.logOutput("  |--Best Algorithm:" + selectedModelMap.head._1 + "\n")
      logger.logOutput("  |--Accuracy:" + fm4d.format( 100 * selectedModelMap.head._2._3) + "%\n")
      logger.logOutput("  |--Total Time:" + (TotalTime1 / 1000.0).toString + "\n")
      logger.logOutput("  |--Best Parameters:\n")
      logger.logOutput("          " + plist + "\n")
      logger.printLine()
      logger.logLine()
      //logger.close()


      //logger.logOutput("Max Time:" + t + ", Total Time:" +  (TotalTime1 / 1000.0).toString + ", eta:" + eta+  ", Accuracy:" + selectedModelMap.head._2._3 + ", Parameters:(" + plist  + ") \n"  )
      return selectedModelMap.head
    }
    else
      {
        val Endtime = new java.util.Date().getTime
        val TotalTime1 = Endtime - StartingTime.getTime
        logger.printLine()
        logger.logLine()
        println("  =>Result:")
        println("  |--Time out before train any model" )
        println("  |--Total Time:" + (TotalTime1 / 1000.0).toString)

        logger.logOutput("  =>Result:\n")
        logger.logOutput("  |--Time out before train any model" )
        logger.logOutput("  |--Total Time:" + (TotalTime1 / 1000.0).toString + "\n")

        logger.printLine()
        logger.logLine()
        //logger.close()

        return null
      }
  }


}
