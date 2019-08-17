package org.apache.spark.ml.tuning

import java.util.Date

import org.apache.spark.annotation.Since
import org.apache.spark.ml.Model
import org.apache.spark.ml.classification._
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.param.shared.HasParallelism
import org.apache.spark.ml.util.Identifiable
import org.apache.spark.sql.functions.{col, min}
import org.apache.spark.sql._
import org.apache.spark.util.ThreadUtils
import org.dsmartml.{ClassifiersManager}

import scala.collection.immutable.ListMap
import scala.concurrent.Future

class RandomSearchManager (ParamNumber:Int,
                           MaxTime:Int,
                           MaxResource:Int,
                           sp:SparkSession,
                           featureCol:String = "features" ,
                           TargetCol:String = "y" ,
                           Parallelism:Int = 3,
                           seed:Long=1234,
                           override val uid: String = Identifiable.randomUID("rs") ) extends HasParallelism {
  var bestParam: ParamMap = null
  var bestModel: Model[_] = null
  var bestmetric: Double = 0.0
  var bestClassifier = ""
  def setParallelism(value: Int): this.type = set(parallelism, value)
  setParallelism(Parallelism)
  var StartingTime: Date = new Date()

  val formatter = java.text.NumberFormat.getInstance
  formatter.setMaximumFractionDigits(2)

  def IsTimeOut(): Boolean = {
    if (getRemainingTime() == 0)
      return true
    else
      return false
  }

  def getRemainingTime(): Long = {
    var rem: Long = (MaxTime * 1000) - (new Date().getTime - StartingTime.getTime())
    //println("-> Remaining Time:" + rem)
    if (rem < 0)
      rem = 0
    return rem
  }

  def fit(dataset: Dataset[_]): Model[_] = {
    if (!IsTimeOut()) {
      //val pwLog = new PrintWriter(new File(logpath ))
      val res = Search(dataset , sp)
      bestParam  = res._1
      bestModel= res._2._2
      bestmetric  = res._2._1
      // return RandomSearch mode (with: best model, best parameters and its evaluation metric)
      return bestModel
    }
    else
      return null
  }

  def Search(dataset: Dataset[_], spark: SparkSession): (ParamMap, (Double, Model[_]))  = {

    val max_Resource = MaxResource
    val paramNumbers = ParamNumber
    val ParamStep = Parallelism
    var RemainingParamCount = 0
    var curParamNumber = 0
    var currentResult = ListMap[ParamMap, (Double, Model[_], String)]()
    val featurecolumns = dataset.columns.filter(c => c != TargetCol)
    val nr_features: Int = featurecolumns.length
    var nr_classes = dataset.groupBy(TargetCol).count().collect().toList.length
    val hasNegativeFeatures = HasNegativeValue(dataset.toDF(), nr_features, nr_classes, TargetCol)
    StartingTime = new Date()

    // Create Classifiers Manager
    val ClassifierMgr = new ClassifiersManager(spark, nr_features, nr_classes)

    var bestParamMap: ParamMap = null
    var bestModel: Model[_] = null
    var bestaccur = 0.0
    var classifer_name = ""
    var parametercount = 0
    var Index = 0


    for (c <- ClassifiersManager.classifiersLsit) {
      RemainingParamCount = ParamNumber
      Index = 0
      var p = ClassifierMgr.getRandomParameters(c, paramNumbers)
      if (IsTimeOut())
        {
          println("Time Out.....")
        }
      else if(nr_classes > 2 && Array(4, 6).contains(ClassifiersManager.classifiersLsit.indexOf(c)))
        {
          println("LinearSVC & GBT not work with more than two classes")
        }
      else if(hasNegativeFeatures == true && ClassifiersManager.classifiersLsit.indexOf(c) == 5)
        {
          println("NaiveBayes not work with negative data")
        }
      else if(ClassifiersManager.classifiersLsit.indexOf(c) == 3)
        {
          println("Multilayer not working")
        }
      else
      {
        println(c)

        while ( RemainingParamCount > 0 && !IsTimeOut() ) {

           if (RemainingParamCount > ParamStep) {
             parametercount = ParamStep
             RemainingParamCount = RemainingParamCount - ParamStep
           }
          else {
             parametercount = RemainingParamCount
             RemainingParamCount = 0
          }

          var arr = new Array[ParamMap](ParamStep)
          for( i <- 0 until  parametercount)
            {
              arr(i) = p(i + Index)
            }
          Index = Index + parametercount

          println( " -- Classifier:" + c + " , Parm Count:" + arr.size)
          println(arr.toSeq.toString())
          var res = learn(dataset, arr, c, ClassifierMgr)
          println(" -- Accuracy:" + res._2._1)
          if (bestaccur < res._2._1) {
            bestaccur = res._2._1
            bestParamMap = res._1
            bestModel = res._2._2
            classifer_name = c
            bestClassifier = c


            //currentResult += ( res._1 -> ( res._2._1 , res._2._2 , c))
          }
        }
      }
    }


    if (classifer_name == "RandomForestClassifier")
      return (bestParamMap, (bestaccur, bestModel.asInstanceOf[RandomForestClassificationModel]))
    else if (classifer_name == "LogisticRegression")
      return (bestParamMap, (bestaccur, bestModel.asInstanceOf[LogisticRegressionModel]))
    else if (classifer_name== "DecisionTreeClassifier")
      return (bestParamMap, (bestaccur, bestModel.asInstanceOf[DecisionTreeClassificationModel]))
    else if (classifer_name == "MultilayerPerceptronClassifier")
      return (bestParamMap, (bestaccur, bestModel.asInstanceOf[MultilayerPerceptronClassificationModel]))
    else if (classifer_name == "LinearSVC")
      return (bestParamMap, (bestaccur, bestModel.asInstanceOf[LinearSVCModel]))
    else if (classifer_name == "NaiveBayes")
      return (bestParamMap, (bestaccur, bestModel.asInstanceOf[NaiveBayesModel]))
    else if (classifer_name == "GBTClassifier")
      return (bestParamMap, (bestaccur, bestModel.asInstanceOf[GBTClassificationModel]))
    else if (classifer_name == "LDA")
      return (bestParamMap, (bestaccur, bestModel.asInstanceOf[LDAModel]))
    else
      return (bestParamMap, (bestaccur, bestModel.asInstanceOf[QDAModel]))


  }

  def HasNegativeValue(df: DataFrame, nr_features: Int, nr_classes: Int, TargetCol: String): Boolean = {
    /*
    val iterationcolumns = 1000
    var cond: Column = null
    var columncounter = 0
    // Min Value
    var ColumnMinValMap = Map[String, Double]()
    var MinValueRow: Row = null

    var l = nr_features / iterationcolumns
    var currcol: Array[String] = null
    var features: Array[String] = df.columns.filter(c => c != TargetCol)

    for (c <- 0 to l) {
      currcol = features.slice(c * iterationcolumns, (c * iterationcolumns) + (iterationcolumns))
      if (currcol.length > 0) {
        // missing values
        cond = currcol.map(x => col(x).isNull).reduce(_ || _)

        //Min
        MinValueRow = df.select(currcol.map(c => min(col(c)).alias(c)): _*).collect()(0)

        for (cc <- currcol) {
          ColumnMinValMap += (cc -> MinValueRow(columncounter).asInstanceOf[Number].doubleValue())
          columncounter = columncounter + 1
        }
        columncounter = 0
      }

    }

    if (ColumnMinValMap.values.toArray.filter(d => d < 0).length > 0)
      return true
    else
      return false
  */
    return true
  }

  def learn(dataset: Dataset[_], param: Array[ParamMap] , ClassifierName:String , ClassifierMgr:ClassifiersManager): (ParamMap, (Double, Model[_])) = {

    try {
      val est = ClassifierMgr.ClassifiersMap(ClassifierName)
      val eval = ClassifierMgr.evaluator
      val epm = param.toList

      // Create execution context based on $(parallelism)
      val executionContext = getExecutionContext

      val Array(trainingDataset, validationDataset) = dataset.randomSplit(Array(0.8, 0.2), seed)

      // cache data
      trainingDataset.cache()
      validationDataset.cache()

      //Map to save the result
      var iterResultMap = collection.mutable.Map[ParamMap, (Double, Model[_])]()

      val metricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
        Future[Double] {
          //println(paramMap.toString())
          //println( "Remaining Time 1:" + getRemainingTime())
          val model = est.fit(trainingDataset, paramMap).asInstanceOf[Model[_]]
          //println( "Remaining Time 2:" + getRemainingTime())
          //if (collectSubModelsParam) {
          //  subModels.get(paramIndex) = model
          //}
          // TODO: duplicate evaluator to take extra params from input
          val metric = eval.evaluate(model.transform(validationDataset, paramMap))

          //pw.write(" Parameters:" + paramMap.toSeq.toString() + ", Metric:" + metric + "\n")
          //println("-- -- Metric:" + metric)
          //println("paramMap:" + paramMap.toString())
          if (ClassifierName == "RandomForestClassifier")
            iterResultMap += (paramMap -> (metric, model.asInstanceOf[RandomForestClassificationModel]))
          else if (ClassifierName == "LogisticRegression")
            iterResultMap += (paramMap -> (metric, model.asInstanceOf[LogisticRegressionModel]))
          else if (ClassifierName== "DecisionTreeClassifier")
            iterResultMap += (paramMap -> (metric, model.asInstanceOf[DecisionTreeClassificationModel]))
          else if (ClassifierName == "MultilayerPerceptronClassifier")
            iterResultMap += (paramMap -> (metric, model.asInstanceOf[MultilayerPerceptronClassificationModel]))
          else if (ClassifierName == "LinearSVC")
            iterResultMap += (paramMap -> (metric, model.asInstanceOf[LinearSVCModel]))
          else if (ClassifierName == "NaiveBayes")
            iterResultMap += (paramMap -> (metric, model.asInstanceOf[NaiveBayesModel]))
          else if (ClassifierName == "GBTClassifier")
            iterResultMap += (paramMap -> (metric, model.asInstanceOf[GBTClassificationModel]))
          else if (ClassifierName == "LDA")
            iterResultMap += (paramMap -> (metric, model.asInstanceOf[LDAModel]))
          else
            iterResultMap += (paramMap -> (metric, model.asInstanceOf[QDAModel]))

          //println("     - "+ ClassifierName +" Accuracy:" + metric )
          metric
        }(executionContext)
      }


      import scala.concurrent.duration._
      val duration = Duration(getRemainingTime(), MILLISECONDS)

      // Wait for all metrics to be calculated
      try {
      //println( "Remaining Time before:" + (getRemainingTime() / 1000.0))
        val metrics = metricFutures.map(ThreadUtils.awaitResult(_, duration)) //Duration.Inf))
      //println( "Remaining Time after:" + (getRemainingTime() / 1000.0))
      } catch {
        case ex: Exception => println("      --TimeOut:==>" + ex.getMessage)
         //println(ex.getStackTrace().toString)
      }
      var sortedIterResultMap =
        if (eval.isLargerBetter)
          ListMap(iterResultMap.toSeq.sortWith(_._2._1 > _._2._1): _*)
        else
          ListMap(iterResultMap.toSeq.sortWith(_._2._1 < _._2._1): _*)

      var bestaccur = 0.0
      var bestParamMap :ParamMap= null
      var bestModel: Model[_] = null
      for ( x <- iterResultMap)
      {
        if (bestaccur < x._2._1)
        {
          bestaccur = x._2._1
          bestParamMap = x._1
          bestModel = x._2._2
        }
      }

      // Unpersist training & validation set once all metrics have been produced
      trainingDataset.unpersist()
      validationDataset.unpersist()


      //println("     ------ best is " + sortedIterResultMap.head._2 + "-----")
      if (ClassifierName == "RandomForestClassifier")
        return (bestParamMap -> (bestaccur , bestModel.asInstanceOf[RandomForestClassificationModel]))
      else if (ClassifierName == "LogisticRegression")
        return (bestParamMap -> (bestaccur , bestModel.asInstanceOf[LogisticRegressionModel]))
      else if (ClassifierName== "DecisionTreeClassifier")
        return (bestParamMap -> (bestaccur , bestModel.asInstanceOf[DecisionTreeClassificationModel]))
      else if (ClassifierName == "MultilayerPerceptronClassifier")
        return (bestParamMap -> (bestaccur , bestModel.asInstanceOf[MultilayerPerceptronClassificationModel]))
      else if (ClassifierName == "LinearSVC")
        return (bestParamMap -> (bestaccur , bestModel.asInstanceOf[LinearSVCModel]))
      else if (ClassifierName == "NaiveBayes")
        return (bestParamMap -> (bestaccur , bestModel.asInstanceOf[NaiveBayesModel]))
      else if (ClassifierName == "GBTClassifier")
        return (bestParamMap -> (bestaccur , bestModel.asInstanceOf[GBTClassificationModel]))
      else if (ClassifierName == "LDA")
        return (bestParamMap -> (bestaccur , bestModel.asInstanceOf[LDAModel]))
      else
        return (bestParamMap -> (bestaccur , bestModel.asInstanceOf[QDAModel]))


    } catch {
      case ex: Exception =>
        println("Exception (Hyperband - " + ClassifierName + "- learn): " + ex.getMessage)
        ex.printStackTrace()
        return null
    }
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): RandomSearch = {
    val copied = defaultCopy(extra).asInstanceOf[RandomSearch]

    copied
  }
}
