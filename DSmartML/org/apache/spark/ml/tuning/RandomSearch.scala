package org.apache.spark.ml.tuning
//package org.dsmartml


import java.util.{Date, Locale, List => JList}

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{Evaluator, MulticlassClassificationEvaluator}
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasCollectSubModels, HasParallelism}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.{col, min}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql._
import org.apache.spark.util.ThreadUtils
import org.dsmartml._
import org.json4s.DefaultFormats

import scala.collection.JavaConverters._
import scala.concurrent.Future
import scala.language.existentials

//import org.dsmartml.Logger

import scala.collection.immutable.ListMap

/**
  * Parameters for [[RandomSearch]] and [[RandomSearchModel]].
  */
trait RandomSearchParams extends ValidatorParams {
  /**
    * controls the Number of
    * configurations of hyperparameters
    * Default: 20
    * @group param
    */
  val ParamNumber: IntParam = new IntParam(this, "ParamNumber",
    "controls the number of random parameterss", ParamValidators.inRange(1, 10000))

  /** @group getParam */
  def getParamNumber: Double = $(ParamNumber)

  setDefault(ParamNumber -> 20)

  /**
    * the maximum amount of resource that can
    * be allocated to a single configuration
    * Default: 1
    *
    * @group param
    */
  val maxResource: IntParam = new IntParam(this, "maxResource",
    "the maximum amount of resource that can\nbe allocated to a single configuration", ParamValidators.inRange(1, 100))

  /** @group getParam */
  def getMaxResource: Double = $(maxResource)

  setDefault(maxResource -> 1)


  /**
    * log file path
    *
    * @group param
    */
  val logFilePath: Param[String] = new Param[String](this, "logFilePath","Log to File path")

  /** @group getParam */
  def getLogFilePath: String = $(logFilePath)

  setDefault(logFilePath -> "/home")

  /**
    * Target Column name
    *
    * @group param
    */
  val TargetColumn: Param[String] = new Param[String](this, "TargetColumn","Target Column name")

  /** @group getParam */
  def getTargetColumn: String = $(TargetColumn)

  setDefault(TargetColumn -> "y")


  /**
    * should i log to file
    *
    * @group param
    */
  val logToFile: BooleanParam = new BooleanParam(this, "logToFile"," should i Log to File")

  /** @group getParam */
  def getLogToFile: Boolean = $(logToFile)

  setDefault(logToFile -> false)




  /**
    * Classifier Name
    *
    * @group param
    */
  val ClassifierName: Param[String] = new Param[String](this, "logFilePath","Log to File path")

  /** @group getParam */
  def getclassifiername: String = $(ClassifierName)
  setDefault(ClassifierName -> "RandomForest")



  /**
    * the maximum Time allowed for RandomSearch on this algorithm
    * Default: Inf
    *
    * @group param
    */
  val maxTime: LongParam = new LongParam(this, "maxTime",
    "the maximum amount of Time (in seconds) that can\nbe allocated to RandomSearch algorithm")

  /** @group getParam */
  def getmaxTime: Double = $(maxTime)

  setDefault(maxTime -> 60 * 10) // 10 minutes




}

/**
  * Class Name: RandomSearch
  * Description: this calss represent RandomSearch implementation & it support parallelism & Time Limit
  * @constructor Create a RandomSearch Object all us to do RandomSearch optimization
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/3/2019
  */
class RandomSearch (@Since("1.5.0") override val uid: String)
  extends Estimator[RandomSearchModel]
    with RandomSearchParams with HasParallelism with HasCollectSubModels
    with MLWritable with Logging {

  var bestParam: ParamMap = null
  var bestModel: Model[_] = null
  var bestmetric: Double = 0.0
  var filelog: Logger = null
  var spark: SparkSession = null

  setEstimatorParamMaps(Array(new ParamMap()))
  setEstimator(new RandomForestClassifier())
  setEvaluator(new MulticlassClassificationEvaluator())

  //var ClassifierParamsMapIndexed = Map[Int, ParamMap]()

  @Since("1.5.0")
  def this() = this(Identifiable.randomUID("tvs"))

  /** @group setParam */
  @Since("1.5.0")
  def setEstimator(value: Estimator[_]): this.type = set(estimator, value)

  /** @group setParam */
  @Since("1.5.0")
  def setEstimatorParamMaps(value: Array[ParamMap]): this.type = set(estimatorParamMaps, value)

  /** @group setParam */
  @Since("1.5.0")
  def setEvaluator(value: Evaluator): this.type = set(evaluator, value)

  /** @group setParam */
  @Since("1.5.0")
  def setParamNumber(value: Integer): this.type = set("ParamNumber", value)

  /** @group setParam */
  @Since("1.5.0")
  def setmaxResource(value: Integer): this.type = set("maxResource", value)

  /** @group setParam */
  @Since("2.0.0")
  def setSeed(value: Long): this.type = set(seed, value)

  @Since("2.3.0")
  def setParallelism(value: Int): this.type = set(parallelism, value)

  @Since("2.3.0")
  def setCollectSubModels(value: Boolean): this.type = set(collectSubModels, value)

  @Since("2.3.0")
  def setLogToFile(value: Boolean): this.type = set(logToFile, value)

  @Since("2.3.0")
  def setTargetColumn(value: String): this.type = set(TargetColumn, value)

  @Since("2.3.0")
  def setLogFilePath(value: String): this.type = set(logFilePath, value)

  def setClassifierName(value: String): this.type = set(ClassifierName, value)


  @Since("2.3.0")
  def setmaxTime(value: Long): this.type = set(maxTime, value)

  val StartingTime: Date = new Date()

  def IsTimeOut(): Boolean = {
    if (getRemainingTime() == 0)
      return true
    else
      return false
  }

  def getRemainingTime(): Long = {
    var rem: Long = ($(maxTime) * 1000) - (new Date().getTime - StartingTime.getTime())
    if (rem < 0)
      rem = 0
    return rem
  }

  val formatter = java.text.NumberFormat.getInstance
  formatter.setMaximumFractionDigits(2)


  /**
    * This function Run RandomSearch Algorithm on the data set to get best algorithm (best hyper parameters)
    *
    * @param dataset the input dataset on which we will run the RandomSearch
    * @return RandomSearchModel object (represent best model found based on accuracy and its hyperparameters)
    */
  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): RandomSearchModel = {
    if (!IsTimeOut()) {
      // properities of RandomSearch
      //val est = $(estimator)
      //val eval = $(evaluator)
      //val logpath = $(logFilePath)
      //val shouldLogToFile = $(logToFile)

      val schema = dataset.schema
      transformSchema(schema, logging = true)


      // split the data to training and validation
      //val Array(trainingDataset, validationDataset) = dataset.randomSplit(Array(0.80, 0.20), $(seed))

      //val pwLog = new PrintWriter(new File(logpath ))
      val res = Search(dataset , spark)
      //pwLog.close()


      var bestParam_ : ParamMap = res._1
      var bestModel_ : Model[_] = res._2._2
      var metric_ : Double = res._2._1


      // return RandomSearch mode (with: best model, best parameters and its evaluation metric)
      return new RandomSearchModel(uid, bestModel_, Array(metric_))
    }
    else
      return null
  }

  /**
    * this function check if the dataset has negative values in any column or not
    *
    * @param df          the dataframe that we will do Grid Search on it
    * @param nr_features the number of features in the dataset
    * @param nr_classes  the number of classes in the dataset
    * @param TargetCol   the label column name
    * @return boolean (true = has negative values, false = no negative values)
    */
  def HasNegativeValue(df: DataFrame, nr_features: Int, nr_classes: Int, TargetCol: String): Boolean = {
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

  }


  def Search(dataset: Dataset[_], spark: SparkSession): (ParamMap, (Double, Model[_]))  = {
    val max_Resource = $(maxResource)
    val paramNumbers = $(ParamNumber)
    val shouldLogtoFile = $(logToFile)
    val TargetCol = $(TargetColumn)
    val ParamStep = 5
    var curParamNumber = 0
    var currentResult = ListMap[ParamMap, (Double, Model[_], String)]()


    val featurecolumns = dataset.columns.filter(c => c != TargetCol)
    val nr_features: Int = featurecolumns.length
    var nr_classes = dataset.groupBy(TargetCol).count().collect().toList.length
    val hasNegativeFeatures = HasNegativeValue(dataset.toDF(), nr_features, nr_classes, TargetCol)


    // Create Classifiers Manager
    val ClassifierMgr = new ClassifiersManager(spark, nr_features, nr_classes)

    var bestParamMap: ParamMap = null
    var bestModel: Model[_] = null
    var bestaccur = 0.0
    var classifer_name = ""

    for (c <- ClassifiersManager.classifiersLsit) {
      if (!IsTimeOut() &&
        (nr_classes == 2 && Array(4, 6).contains(ClassifiersManager.classifiersLsit.indexOf(c))) &&
        (nr_classes >= 2 && Array(0, 1, 2, 3).contains(ClassifiersManager.classifiersLsit.indexOf(c))) &&
        (hasNegativeFeatures == false && ClassifiersManager.classifiersLsit.indexOf(c) == 5)
      ) {
        var p = ClassifierMgr.getRandomParameters(c, ParamStep)
        var res = learn(dataset, p, c , ClassifierMgr)
        if (bestaccur < res._2._1) {
          bestaccur = res._2._1
          bestParamMap = res._1
          bestModel = res._2._2
          classifer_name = c
          //currentResult += ( res._1 -> ( res._2._1 , res._2._2 , c))
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


  /**
    * this function ML algorithm training for a set of hyper parameter configuration, the training could be parallelized (if the cluster can has free nodes)
    *
    * @param dataset The input dataset on which we will run the RandomSearch
    * @param param   the yperparameters
    * @param r
    * @return
    */
  def learn(dataset: Dataset[_], param: Array[ParamMap] , ClassifierName:String , ClassifierMgr:ClassifiersManager): (ParamMap, (Double, Model[_])) = {
    try {
      val schema = dataset.schema
      transformSchema(schema, logging = true)
      val est = ClassifierMgr.ClassifiersMap(ClassifierName)
      val eval = ClassifierMgr.evaluator
      val epm = param.toList
      val shouldLogtoFile = $(logToFile)


      // Create execution context based on $(parallelism)
      val executionContext = getExecutionContext

      val Array(trainingDataset, validationDataset) = dataset.randomSplit(Array(0.8, 0.2), $(seed))

      // cache data
      trainingDataset.cache()
      validationDataset.cache()

      //Map to save the result
      var iterResultMap = collection.mutable.Map[ParamMap, (Double, Model[_])]()

      // Fit models in a Future for training in parallel
      logDebug(s"Train split with multiple sets of parameters.")
      val metricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
        Future[Double] {
          val model = est.fit(trainingDataset, paramMap).asInstanceOf[Model[_]]

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

          //println("     - Accuracy:" + metric )
          metric
        }(executionContext)
      }


      import scala.concurrent.duration._
      val duration = Duration(getRemainingTime(), MILLISECONDS)

      // Wait for all metrics to be calculated
      try {
        val metrics = metricFutures.map(ThreadUtils.awaitResult(_, duration)) //Duration.Inf))
      } catch {
        case ex: Exception => println("      --TimeOut:==>" + ex.getMessage)
          println(ex.getStackTrace())
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
  override def transformSchema(schema: StructType): StructType = transformSchemaImpl(schema)

  @Since("1.5.0")
  override def copy(extra: ParamMap): RandomSearch = {
    val copied = defaultCopy(extra).asInstanceOf[RandomSearch]
    if (copied.isDefined(estimator)) {
      copied.setEstimator(copied.getEstimator.copy(extra))
    }
    if (copied.isDefined(evaluator)) {
      copied.setEvaluator(copied.getEvaluator.copy(extra))
    }
    copied
  }

  @Since("2.0.0")
  override def write: MLWriter = new RandomSearch.RandomSearchWriter(this)



}


@Since("2.0.0")
object RandomSearch extends MLReadable[RandomSearch] {

  @Since("2.0.0")
  override def read: MLReader[RandomSearch] = new RandomSearchReader

  @Since("2.0.0")
  override def load(path: String): RandomSearch = super.load(path)

  private[RandomSearch] class RandomSearchWriter(instance: RandomSearch)
    extends MLWriter {

    ValidatorParams.validateParams(instance)

    override protected def saveImpl(path: String): Unit =
      ValidatorParams.saveImpl(path, instance, sc)
  }

  private class RandomSearchReader extends MLReader[RandomSearch] {

    /** Checked against metadata when loading model */
    private val className = classOf[RandomSearch].getName

    override def load(path: String): RandomSearch = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val tvs = new RandomSearch(metadata.uid)
        .setEstimator(estimator)
        .setEvaluator(evaluator)
       // .setEstimatorParamMaps(estimatorParamMaps)
      //DefaultParamsReader.getAndSetParams(tvs, metadata, skipParams = Option(List("estimatorParamMaps")))
      tvs
    }
  }
}



/**
  * Model from train RandomSearch.
  *
  * @param uid Id.
  * @param bestModel Estimator determined best model.
  * @param validationMetrics Evaluated validation metrics.
  */
@Since("1.5.0")
class RandomSearchModel private[ml] (
                                      @Since("1.5.0") override val uid: String,
                                      @Since("1.5.0") val bestModel: Model[_],
                                      @Since("1.5.0") val validationMetrics: Array[Double]) extends Model[RandomSearchModel] with RandomSearchParams with MLWritable {

  /** A Python-friendly auxiliary constructor. */
  private[ml] def this(uid: String, bestModel: Model[_], validationMetrics: JList[Double]) = {
    this(uid, bestModel, validationMetrics.asScala.toArray)
  }

  private var _subModels: Option[Array[Model[_]]] = None

  private[tuning] def setSubModels(subModels: Option[Array[Model[_]]])
  : RandomSearchModel = {
    _subModels = subModels
    this
  }

  /**
    * @return submodels represented in array. The index of array corresponds to the ordering of
    *         estimatorParamMaps
    * @throws IllegalArgumentException if subModels are not available. To retrieve subModels,
    *         make sure to set collectSubModels to true before fitting.
    */
  @Since("2.3.0")
  def subModels: Array[Model[_]] = {
    require(_subModels.isDefined, "subModels not available, To retrieve subModels, make sure " +
      "to set collectSubModels to true before fitting.")
    _subModels.get
  }

  @Since("2.3.0")
  def hasSubModels: Boolean = _subModels.isDefined

  @Since("2.0.0")
  override def transform(dataset: Dataset[_]): DataFrame = {
    transformSchema(dataset.schema, logging = true)
    bestModel.transform(dataset)
  }

  @Since("1.5.0")
  override def transformSchema(schema: StructType): StructType = {
    bestModel.transformSchema(schema)
  }

  @Since("1.5.0")
  override def copy(extra: ParamMap): RandomSearchModel = {
    val copied = new RandomSearchModel (
      uid,
      bestModel.copy(extra).asInstanceOf[Model[_]],
      validationMetrics.clone()
    ).setSubModels(RandomSearchModel.copySubModels(_subModels))
    copyValues(copied, extra).setParent(parent)
  }

  @Since("2.0.0")
  override def write: RandomSearchModel.RandomSearchModelWriter = {
    new RandomSearchModel.RandomSearchModelWriter(this)
  }
}

@Since("2.0.0")
object RandomSearchModel extends MLReadable[RandomSearchModel] {

  private[RandomSearchModel] def copySubModels(subModels: Option[Array[Model[_]]])
  : Option[Array[Model[_]]] = {
    subModels.map(_.map(_.copy(ParamMap.empty).asInstanceOf[Model[_]]))
  }

  @Since("2.0.0")
  override def read: MLReader[RandomSearchModel] = new RandomSearchModelReader

  @Since("2.0.0")
  override def load(path: String): RandomSearchModel = super.load(path)

  /**
    * Writer for TrainValidationSplitModel.
    * @param instance TrainValidationSplitModel instance used to construct the writer
    *
    * TrainValidationSplitModel supports an option "persistSubModels", with possible values
    * "true" or "false". If you set the collectSubModels Param before fitting, then you can
    * set "persistSubModels" to "true" in order to persist the subModels. By default,
    * "persistSubModels" will be "true" when subModels are available and "false" otherwise.
    * If subModels are not available, then setting "persistSubModels" to "true" will cause
    * an exception.
    */
  @Since("2.3.0")
  final class RandomSearchModelWriter private[tuning] (
                                                        instance: RandomSearchModel) extends MLWriter {

    ValidatorParams.validateParams(instance)

    override protected def saveImpl(path: String): Unit = {
      val persistSubModelsParam = optionMap.getOrElse("persistsubmodels",
        if (instance.hasSubModels) "true" else "false")

      require(Array("true", "false").contains(persistSubModelsParam.toLowerCase(Locale.ROOT)),
        s"persistSubModels option value ${persistSubModelsParam} is invalid, the possible " +
          "values are \"true\" or \"false\"")
      val persistSubModels = persistSubModelsParam.toBoolean

      import org.json4s.JsonDSL._
      val extraMetadata = ("validationMetrics" -> instance.validationMetrics.toSeq) ~
        ("persistSubModels" -> persistSubModels)
      ValidatorParams.saveImpl(path, instance, sc, Some(extraMetadata))
      val bestModelPath = new Path(path, "bestModel").toString
      instance.bestModel.asInstanceOf[MLWritable].save(bestModelPath)
      /*if (persistSubModels) {
        require(instance.hasSubModels, "When persisting tuning models, you can only set " +
          "persistSubModels to true if the tuning was done with collectSubModels set to true. " +
          "To save the sub-models, try rerunning fitting with collectSubModels set to true.")
        val subModelsPath = new Path(path, "subModels")
        for (paramIndex <- 0 until instance.getEstimatorParamMaps.length) {
          val modelPath = new Path(subModelsPath, paramIndex.toString).toString
          instance.subModels(paramIndex).asInstanceOf[MLWritable].save(modelPath)
        }
      }*/
    }
  }

  private class RandomSearchModelReader extends MLReader[RandomSearchModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[RandomSearchModel].getName

    override def load(path: String): RandomSearchModel = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val bestModelPath = new Path(path, "bestModel").toString
      val bestModel = DefaultParamsReader.loadParamsInstance[Model[_]](bestModelPath, sc)
      val validationMetrics = (metadata.metadata \ "validationMetrics").extract[Seq[Double]].toArray
      val persistSubModels = (metadata.metadata \ "persistSubModels")
        .extractOrElse[Boolean](false)

      val subModels: Option[Array[Model[_]]] =  None /*if (persistSubModels) {
        val subModelsPath = new Path(path, "subModels")
        val _subModels = Array.fill[Model[_]](estimatorParamMaps.length)(null)
        for (paramIndex <- 0 until estimatorParamMaps.length) {
          val modelPath = new Path(subModelsPath, paramIndex.toString).toString
          _subModels(paramIndex) =
            DefaultParamsReader.loadParamsInstance(modelPath, sc)
        }
        Some(_subModels)
      } else None */

      val model = new RandomSearchModel(metadata.uid, bestModel, validationMetrics)
        .setSubModels(subModels)
      model.set(model.estimator, estimator)
        .set(model.evaluator, evaluator)
        //.set(model.estimatorParamMaps, estimatorParamMaps)
      //DefaultParamsReader.getAndSetParams(model, metadata,skipParams = Option(List("estimatorParamMaps")))
      model
    }
  }
}






