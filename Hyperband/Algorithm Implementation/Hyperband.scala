package org.apache.spark.ml.tuning

import java.util.{Locale, List => JList}

import scala.collection.JavaConverters._
import scala.concurrent.Future
import scala.concurrent.duration.Duration
import scala.language.existentials
import org.apache.hadoop.fs.Path
import org.json4s.DefaultFormats
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param.{BooleanParam, DoubleParam, IntParam, Param, ParamMap, ParamValidators}
import org.apache.spark.ml.param.shared.{HasCollectSubModels, HasParallelism}
import org.apache.spark.ml.util._
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.sql.types.StructType
import org.apache.spark.util.ThreadUtils
import scala.collection.immutable.ListMap
import java.io._




/**
  * Params for [[Hyperband]] and [[HyperbandModel]].
  */
trait HyperbandParams extends ValidatorParams {
  /**
    * controls the proportion of
    * configurations discarded in each round of Hyperband
    * Default: 3
    *
    * @group param
    */
  val eta: IntParam = new IntParam(this, "eta",
    "controls the proportion of\nconfigurations discarded in each round of Hyperband", ParamValidators.inRange(2, 10))

  /** @group getParam */
  def getEta: Double = $(eta)

  setDefault(eta -> 3)

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
    * should i log to file
    *
    * @group param
    */
  val logToFile: BooleanParam = new BooleanParam(this, "logToFile"," should i Log to File")

  /** @group getParam */
  def getLogToFile: Boolean = $(logToFile)

  setDefault(logToFile -> false)


}

class Hyperband (@Since("1.5.0") override val uid: String)
  extends Estimator[HyperbandModel]
    with HyperbandParams with HasParallelism with HasCollectSubModels
    with MLWritable with Logging {


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
  def setEta(value: Integer): this.type = set("eta", value)

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
  def setLogFilePath(value: String): this.type = set(logFilePath, value)


  /*
  * Run Hyperband Algorithm on the data set to get best algorithm (best hyper parameters)
  * */
  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): HyperbandModel = {

    // properities of Hyperband
    val est = $(estimator)
    val eval = $(evaluator)
    val logpath = $(logFilePath)
    val shouldLogToFile = $(logToFile)

    val schema = dataset.schema
    transformSchema(schema, logging = true)


    val pwLog = new PrintWriter(new File(logpath ))
    val res = hyberband(dataset, pwLog)
    pwLog.close()

    // split the data to training and validation
    val Array(trainingDataset, validationDataset) =
      dataset.randomSplit(Array(0.75, 0.25), $(seed))

    // get the best parameters returned by the Hyperband
    var bestParam =  ListMap(res.toSeq.sortWith(_._2 > _._2):_*).take(1).keys.head

    // train model using best parameters
    val bestModel = est.fit(dataset, bestParam ).asInstanceOf[Model[_]]

    // evaluate the best Model
    val metric = eval.evaluate(bestModel.transform(validationDataset, bestParam))

    // return Hyperband mode (with: best model, best parameters and its evaluation metric)
    new HyperbandModel(uid, bestModel, Array(metric))
  }


  /*
  * Run Hyperband algorithm
  * */
  def hyberband (dataset: Dataset[_] ,  pwLog: PrintWriter ):ListMap[ParamMap, Double] = {

    // properities of hyperband
    val eeta = $(eta)
    val max_Resource = $(maxResource)
    val shouldLogtoFile = $(logToFile)

    if(shouldLogtoFile)
      pwLog.write("---------------------Start Hyperband--------------------------------------\n")

    // List of parameters for each sh iteration
    var currentResult = ListMap[ParamMap, Double]()

    // Log eta
    var logeta =  (x:Double) =>  Math.log(x)/Math.log(eeta)

    // number of unique executions of Successive Halving (minus one)
    var s_max = math.round(logeta(max_Resource))

    //Budget (without reuse) per execution of Succesive Halving (n,r)
    var B = (s_max+1) * max_Resource

    if(shouldLogtoFile) {
      pwLog.write("S max = " + s_max + "\n")
      pwLog.write("--------------------------------------------------------------------------\n")
    }

    // loop (number of successive halving, with different number of hyper-parameters configurations)
    // incearsing number of configuration mean decreasing the resource per each configuration
    for( s <- s_max  to 0 by -1) {
      //initial number of configurations
      var tmp = math.ceil((B / max_Resource / (s + 1)))
      var n = math.round(tmp * math.pow(eeta, s))
      //initial number of resources to run configurations for
      var r = max_Resource * math.pow(eeta, (-s))

      val rsult = sh(dataset,n.toInt,r,s.toInt , pwLog)
      rsult.foreach { case (p,m) =>
        currentResult += (p -> m)
      }
    }
    currentResult
  }


  /*
  * run successive halving algorithm
  * */
  def sh (dataset: Dataset[_] , n: Int, r: Double , s: Int, pw: PrintWriter): ListMap[ParamMap, Double] = {

    // properities
    val shouldLogtoFile = $(logToFile)
    val eeta:Double = $(eta)
    val epm = $(estimatorParamMaps)
    val rand = scala.util.Random

    if(shouldLogtoFile)
      pw.write("========================Start SH with (s="+s+",n=" + n + ",r=" + r + ")=========================\n")

    // select random n hyper parameters configuration
    var iterParams = rand.shuffle(epm.toList).take(n).toArray

    // list to save hyper parameter configuration during the sh iteration
    var currentResult = ListMap[ParamMap, Double]()

    for ( i <-  0 to (s ).toInt)
    {
      //Run each of the n_i configs for r_i iterations and keep best n_i/eta
      var n_i = n* math.pow(eeta,(-i))
      var r_i:Double =   r * math.pow(eeta,(i))
      var resultParam =
        if ( i == 0)
          currentResult = learn(dataset,iterParams,r_i,pw)
        else
          currentResult = learn(dataset,currentResult.take(n_i.toInt).keys.toArray,r_i,pw)

    }
    // return best hyper parameters for this SH run
    ListMap(currentResult.toSeq.sortWith(_._2 > _._2):_*).take(1)

  }

  /*
  ML algorithm training for a set of hyper parameter configuration
  the training could be parallelized (if the cluster can has free nodes)
  * */
  def learn(dataset: Dataset[_] , param : Array[ParamMap] , r : Double, pw: PrintWriter): ListMap[ParamMap , Double] = {
    val schema = dataset.schema
    transformSchema(schema, logging = true)
    val est = $(estimator)
    val eval = $(evaluator)
    val epm = param
    val shouldLogtoFile = $(logToFile)

    if(shouldLogtoFile)
      pw.write("---------------------Start Learn with (r=" + r + ")-------------------------------\n")
    // Create execution context based on $(parallelism)
    val executionContext = getExecutionContext

    val Array(partation, restofdata) =
      dataset.randomSplit(Array(r/100, 1-(r/100)), $(seed))

    val Array(trainingDataset, validationDataset) =
      partation.randomSplit(Array(0.75, 0.25), $(seed))

    // cache data
    trainingDataset.cache()
    validationDataset.cache()

    // if i should keep each model or not
    val collectSubModelsParam = $(collectSubModels)
    var subModels: Option[Array[Model[_]]] = if (collectSubModelsParam) {
      Some(Array.fill[Model[_]](epm.length)(null))
    } else None


    var iterResultMap = collection.mutable.Map[ParamMap, Double]()

    // Fit models in a Future for training in parallel
    logDebug(s"Train split with multiple sets of parameters.")
    val metricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
      Future[Double] {
        val model = est.fit(trainingDataset, paramMap).asInstanceOf[Model[_]]

        if (collectSubModelsParam) {
          subModels.get(paramIndex) = model
        }
        // TODO: duplicate evaluator to take extra params from input
        val metric = eval.evaluate(model.transform(validationDataset, paramMap))
        logDebug(s"Got metric $metric for model trained with $paramMap.")

        pw.write(" Parameters:" + paramMap.toSeq.toString() + ", Metric:" + metric + "\n")
        iterResultMap += ( paramMap -> metric)
        metric
      } (executionContext)
    }

    var sortedIterResultMap =
      if (eval.isLargerBetter)
        ListMap(iterResultMap.toSeq.sortWith(_._2 > _._2):_*)
      else
        ListMap(iterResultMap.toSeq.sortWith(_._2 < _._2):_*)

    // Wait for all metrics to be calculated
    val metrics = metricFutures.map(ThreadUtils.awaitResult(_, Duration.Inf))

    // Unpersist training & validation set once all metrics have been produced
    trainingDataset.unpersist()
    validationDataset.unpersist()

    sortedIterResultMap
  }




  @Since("1.5.0")
  override def transformSchema(schema: StructType): StructType = transformSchemaImpl(schema)

  @Since("1.5.0")
  override def copy(extra: ParamMap): Hyperband = {
    val copied = defaultCopy(extra).asInstanceOf[Hyperband]
    if (copied.isDefined(estimator)) {
      copied.setEstimator(copied.getEstimator.copy(extra))
    }
    if (copied.isDefined(evaluator)) {
      copied.setEvaluator(copied.getEvaluator.copy(extra))
    }
    copied
  }

  @Since("2.0.0")
  override def write: MLWriter = new Hyperband.HyperbandWriter(this)



}


@Since("2.0.0")
object Hyperband extends MLReadable[Hyperband] {

  @Since("2.0.0")
  override def read: MLReader[Hyperband] = new HyperbandReader

  @Since("2.0.0")
  override def load(path: String): Hyperband = super.load(path)

  private[Hyperband] class HyperbandWriter(instance: Hyperband)
    extends MLWriter {

    ValidatorParams.validateParams(instance)

    override protected def saveImpl(path: String): Unit =
      ValidatorParams.saveImpl(path, instance, sc)
  }

  private class HyperbandReader extends MLReader[Hyperband] {

    /** Checked against metadata when loading model */
    private val className = classOf[Hyperband].getName

    override def load(path: String): Hyperband = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val tvs = new Hyperband(metadata.uid)
        .setEstimator(estimator)
        .setEvaluator(evaluator)
        .setEstimatorParamMaps(estimatorParamMaps)
      DefaultParamsReader.getAndSetParams(tvs, metadata,
        skipParams = Option(List("estimatorParamMaps")))
      tvs
    }
  }
}



/**
  * Model from train Hyperband.
  *
  * @param uid Id.
  * @param bestModel Estimator determined best model.
  * @param validationMetrics Evaluated validation metrics.
  */
@Since("1.5.0")
class HyperbandModel private[ml] (
                                   @Since("1.5.0") override val uid: String,
                                   @Since("1.5.0") val bestModel: Model[_],
                                   @Since("1.5.0") val validationMetrics: Array[Double]) extends Model[HyperbandModel] with HyperbandParams with MLWritable {

  /** A Python-friendly auxiliary constructor. */
  private[ml] def this(uid: String, bestModel: Model[_], validationMetrics: JList[Double]) = {
    this(uid, bestModel, validationMetrics.asScala.toArray)
  }

  private var _subModels: Option[Array[Model[_]]] = None

  private[tuning] def setSubModels(subModels: Option[Array[Model[_]]])
  : HyperbandModel = {
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
  override def copy(extra: ParamMap): HyperbandModel = {
    val copied = new HyperbandModel (
      uid,
      bestModel.copy(extra).asInstanceOf[Model[_]],
      validationMetrics.clone()
    ).setSubModels(HyperbandModel.copySubModels(_subModels))
    copyValues(copied, extra).setParent(parent)
  }

  @Since("2.0.0")
  override def write: HyperbandModel.HyperbandModelWriter = {
    new HyperbandModel.HyperbandModelWriter(this)
  }
}

@Since("2.0.0")
object HyperbandModel extends MLReadable[HyperbandModel] {

  private[HyperbandModel] def copySubModels(subModels: Option[Array[Model[_]]])
  : Option[Array[Model[_]]] = {
    subModels.map(_.map(_.copy(ParamMap.empty).asInstanceOf[Model[_]]))
  }

  @Since("2.0.0")
  override def read: MLReader[HyperbandModel] = new HyperbandModelReader

  @Since("2.0.0")
  override def load(path: String): HyperbandModel = super.load(path)

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
  final class HyperbandModelWriter private[tuning] (
                                                     instance: HyperbandModel) extends MLWriter {

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
      if (persistSubModels) {
        require(instance.hasSubModels, "When persisting tuning models, you can only set " +
          "persistSubModels to true if the tuning was done with collectSubModels set to true. " +
          "To save the sub-models, try rerunning fitting with collectSubModels set to true.")
        val subModelsPath = new Path(path, "subModels")
        for (paramIndex <- 0 until instance.getEstimatorParamMaps.length) {
          val modelPath = new Path(subModelsPath, paramIndex.toString).toString
          instance.subModels(paramIndex).asInstanceOf[MLWritable].save(modelPath)
        }
      }
    }
  }

  private class HyperbandModelReader extends MLReader[HyperbandModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[HyperbandModel].getName

    override def load(path: String): HyperbandModel = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val bestModelPath = new Path(path, "bestModel").toString
      val bestModel = DefaultParamsReader.loadParamsInstance[Model[_]](bestModelPath, sc)
      val validationMetrics = (metadata.metadata \ "validationMetrics").extract[Seq[Double]].toArray
      val persistSubModels = (metadata.metadata \ "persistSubModels")
        .extractOrElse[Boolean](false)

      val subModels: Option[Array[Model[_]]] = if (persistSubModels) {
        val subModelsPath = new Path(path, "subModels")
        val _subModels = Array.fill[Model[_]](estimatorParamMaps.length)(null)
        for (paramIndex <- 0 until estimatorParamMaps.length) {
          val modelPath = new Path(subModelsPath, paramIndex.toString).toString
          _subModels(paramIndex) =
            DefaultParamsReader.loadParamsInstance(modelPath, sc)
        }
        Some(_subModels)
      } else None

      val model = new HyperbandModel(metadata.uid, bestModel, validationMetrics)
        .setSubModels(subModels)
      model.set(model.estimator, estimator)
        .set(model.evaluator, evaluator)
        .set(model.estimatorParamMaps, estimatorParamMaps)
      DefaultParamsReader.getAndSetParams(model, metadata,
        skipParams = Option(List("estimatorParamMaps")))
      model
    }
  }
}






