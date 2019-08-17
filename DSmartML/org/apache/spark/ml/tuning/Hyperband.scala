package org.apache.spark.ml.tuning
//package org.dsmartml


import java.util.{Date, Locale, List => JList}

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.Evaluator
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasCollectSubModels, HasParallelism}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset}
import org.apache.spark.util.ThreadUtils
import org.dsmartml._
import org.json4s.DefaultFormats

import scala.collection.JavaConverters._
import scala.concurrent.Future
import scala.language.existentials

//import org.dsmartml.Logger

import scala.collection.immutable.ListMap

/**
  * Parameters for [[Hyperband]] and [[HyperbandModel]].
  */
trait HyperbandParams extends ValidatorParams {
  /**
    * controls the proportion of
    * configurations discarded in each round of Hyperband
    * Default: 3
    * @group param
    */
  val eta: IntParam = new IntParam(this, "eta",
    "controls the proportion of\nconfigurations discarded in each round of Hyperband", ParamValidators.inRange(2, 100))

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
    * the maximum Time allowed for Hyperband on this algorithm
    * Default: Inf
    *
    * @group param
    */
  val maxTime: LongParam = new LongParam(this, "maxTime",
    "the maximum amount of Time (in seconds) that can\nbe allocated to Hyperband algorithm")

  /** @group getParam */
  def getmaxTime: Double = $(maxTime)

  setDefault(maxTime -> 60 * 10) // 10 minutes




}

/**
  * Class Name: Hyperband
  * Description: this calss represent hyperband implementation & it support parallelism
  * @constructor Create a hyperband Object all us to do hyperband optimization
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/3/2019
  */
class Hyperband (@Since("1.5.0") override val uid: String)
  extends Estimator[HyperbandModel]
    with HyperbandParams with HasParallelism with HasCollectSubModels
    with MLWritable with Logging {

  var bestParam :ParamMap = null
  var bestModel :Model[_] = null
  var bestmetric :Double = 0.0
  var filelog :Logger = null
  var ClassifierParamsMapIndexed =  Map[Int,ParamMap]()
  var ClassifiersMgr: ClassifiersManager = null

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

  def setClassifierName(value: String): this.type = set(ClassifierName, value)


  @Since("2.3.0")
  def setmaxTime(value: Long): this.type = set(maxTime, value)

  val StartingTime:Date = new Date()

  def IsTimeOut(): Boolean =
  {
    if ( getRemainingTime() == 0)
      return true
    else
      return false
  }

  def getRemainingTime(): Long =
  {
    var rem:Long = ($(maxTime) * 1000) - (new Date().getTime - StartingTime.getTime())
    if (rem < 0)
      rem = 0
    return rem
  }

  val formatter = java.text.NumberFormat.getInstance
  formatter.setMaximumFractionDigits(2)




  /**
    * This function Run Hyperband Algorithm on the data set to get best algorithm (best hyper parameters)
    * @param dataset the input dataset on which we will run the hyperband
    * @return hyperbandModel object (represent best model found based on accuracy and its hyperparameters)
    */
  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): HyperbandModel = {
    if (!IsTimeOut()) {
      // properities of Hyperband
      val est = $(estimator)
      val eval = $(evaluator)
      val logpath = $(logFilePath)
      val shouldLogToFile = $(logToFile)

      val schema = dataset.schema
      transformSchema(schema, logging = true)


      // split the data to training and validation
      //val Array(trainingDataset, validationDataset) = dataset.randomSplit(Array(0.80, 0.20), $(seed))

      //val pwLog = new PrintWriter(new File(logpath ))
      val res = hyberband(dataset)
      //pwLog.close()


      /*
      var bestParam_ :ParamMap = null
      var bestModel_ :Model[_] = null
      var metric_ :Double = 0.0

      res.foreach { case (p, (a, m)) =>
        val metric = eval.evaluate(m.transform(validationDataset))
        println(" === Accuracy:" + metric)
          if(metric > metric_){
            bestParam_ = p
            bestModel_ = m
            metric_ = metric


            this.bestParam = bestParam_
            this.bestModel = bestModel_
            this.bestmetric = metric_


          }

      }
      */
        // get the best parameters returned by the Hyperband
      var bestParam:ParamMap = ListMap(res.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1).keys.head

      // train model using best parameters
      //val bestModel = est.fit(trainingDataset, bestParam ).asInstanceOf[Model[_]]
      val bestModel:Model[_] = ListMap(res.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1).values.head._2
      // evaluate the best Model
      //val metric = eval.evaluate(bestModel.transform(validationDataset, bestParam))
      val metric:Double = ListMap(res.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1).values.head._1


      this.bestParam = bestParam
      this.bestModel = bestModel
      this.bestmetric = metric


      // return Hyperband mode (with: best model, best parameters and its evaluation metric)
      return new HyperbandModel(uid, bestModel, Array(metric))
    }
    else
      return null
  }




  /**
    * The Hyperband implemtation
    * @param dataset The input dataset on which we will run the hyperband
    * @return List of parameters for each sh iteration
    */
  def hyberband (dataset: Dataset[_]  ):ListMap[ParamMap, (Double,Model[_])] = {

    //println("start Hyper band with eta = " + $(eta) + ", and max resource = " + $(maxResource) + "%")
    //println("--------------------------------------------------------------------------------")
    // properities of hyperband
    val eeta = $(eta)
    val max_Resource = $(maxResource)
    val shouldLogtoFile = $(logToFile)

    //if(shouldLogtoFile)
    //  pwLog.write("---------------------Start Hyperband--------------------------------------\n")

    // List of parameters for each sh iteration
    var currentResult = ListMap[ParamMap, (Double,Model[_])]()

    // Log eta
    var logeta =  (x:Double) =>  Math.log(x)/Math.log(eeta)

    // number of unique executions of Successive Halving (minus one)
    var s_max = math.round(logeta(max_Resource))
    //println("Number of Successive Halving Sessions = " + (s_max + 1))

    //Budget (without reuse) per execution of Succesive Halving (n,r)
    var B = (s_max+1) * max_Resource

    //if(shouldLogtoFile) {
    //pwLog.write("S max = " + s_max + "\n")
    //pwLog.write("--------------------------------------------------------------------------\n")
    //}

    var firstSH = true
    // loop (number of successive halving, with different number of hyper-parameters configurations)
    // incearsing number of configuration mean decreasing the resource per each configuration
    for( s <- s_max-1  to 0 by -1) {
      if (!IsTimeOut()) {

        //initial number of configurations
        var tmp = math.ceil((B / max_Resource / (s + 1)))
        var n = math.round(tmp * math.pow(eeta, s))
        //println("   - Number of Hyperparameter values to check =" + n)

        //initial number of resources to run configurations for
        var r = max_Resource * math.pow(eeta, (-s))
        //println("   - Initial resource percentage to check =" + r)
        filelog.logException("=================================================================================================\n")
        filelog.logException("     -- Successive Halving Session:" + s + " , with Max Resource = " + formatter.format(r) + " and Hyperparameter values to check  "  + n +"\n")
        println("     -- Successive Halving Session:" + s + " , with Max Resource = " + formatter.format(r) + " and Hyperparameter values to check  "  + n )

        val rsult = sh(dataset, n.toInt, r, s.toInt , firstSH)

        firstSH = false
        if (rsult != null) {
          if ($(ClassifierName) == "RandomForestClassifier")
            rsult.foreach { case (p, (a, m ,pm)) => currentResult += (pm -> (a, m.asInstanceOf[RandomForestClassificationModel])) }
          else if ($(ClassifierName) == "LogisticRegression")
            rsult.foreach { case (p, (a, m ,pm)) => currentResult += (pm -> (a, m.asInstanceOf[LogisticRegressionModel])) }
          else if ($(ClassifierName) == "DecisionTreeClassifier")
            rsult.foreach { case (p, (a, m , pm)) => currentResult += (pm -> (a, m.asInstanceOf[DecisionTreeClassificationModel])) }
          else if ($(ClassifierName) == "MultilayerPerceptronClassifier")
            rsult.foreach { case (p, (a, m , pm)) => currentResult += (pm -> (a, m.asInstanceOf[MultilayerPerceptronClassificationModel])) }
          else if ($(ClassifierName) == "LinearSVC")
            rsult.foreach { case (p, (a, m , pm)) => currentResult += (pm -> (a, m.asInstanceOf[LinearSVCModel])) }
          else if ($(ClassifierName) == "NaiveBayes")
            rsult.foreach { case (p, (a, m , pm)) => currentResult += (pm -> (a, m.asInstanceOf[NaiveBayesModel])) }
          else if ($(ClassifierName) == "GBTClassifier")
            rsult.foreach { case (p, (a, m , pm)) => currentResult += (pm -> (a, m.asInstanceOf[GBTClassificationModel])) }
          else if ($(ClassifierName) == "LDA")
            rsult.foreach { case (p, (a, m, pm)) => currentResult += (pm -> (a, m.asInstanceOf[LDAModel])) }
          else
            rsult.foreach { case (p, (a, m,pm)) => currentResult += (pm -> (a, m.asInstanceOf[QDAModel])) }
        }
      }
      else
        {
          println("     --Time out @ Main Hyperband loop")
          return currentResult
        }
    }
    currentResult
  }




  /**
    * This function run successive halving algorithm for spacific resources on spacific iteration
    * @param dataset The input dataset on which we will run the hyperband
    * @param n number of hyperparameters configuration
    * @param r resources (percentage of Records)
    * @param s number of iterations
    * @return return best hyper parameters for this SH run
    */
  def sh (dataset: Dataset[_] , n: Int, r: Double , s: Int, isFirstSH:Boolean): Map[Int, (Double,Model[_],ParamMap)] = {

    // properities
    val shouldLogtoFile = $(logToFile)
    val eeta:Double = $(eta)
    //val epm = $(estimatorParamMaps)
    //val rand = scala.util.Random

/*

    // select random n hyper parameters configuration
    val rand = new util.Random()
    rand.setSeed(1234 + s)
    val end   = this.ClassifierParamsMapIndexed.toList.size
    var ParamArray:Array[ParamMap] = new Array[ParamMap](n)
    var iter = 0
    var usedIndecies = List[Int]()
    var SelectedParamsMapIndexed =  Map[Int,ParamMap]()

    //in case of all the parameters less than the needed parametr by hyperband first sh
    if( end <= n) {
      SelectedParamsMapIndexed  = this.ClassifierParamsMapIndexed
    }
    else
    {
      while (iter < n) {
        var x = rand.nextInt(end)
        if (!usedIndecies.contains(x)) {
          usedIndecies = x :: usedIndecies
          SelectedParamsMapIndexed += (x -> this.ClassifierParamsMapIndexed(x))
          iter = iter + 1
        }
      }
    }

*/

    var SelectedParamsMapIndexed =  Map[Int,ParamMap]()
    SelectedParamsMapIndexed = ClassifiersMgr.getRandomParametersIndexed( $(ClassifierName), n)

    var s12 = ""
    for ( ps <- SelectedParamsMapIndexed) {
      for (p <- ps._2.toSeq) {
        s12 = p.param.name + ":" + p.value + "," + s12
        //println(s12)
        //filelog.logException(p.param.name + ":" + p.value  + "\n")
      }

      filelog.logException(" -->>" + s12 + "\n")
      s12 = ""
    }

    // list to save hyper parameter configuration during the sh iteration
    var currentResult = ListMap[Int, (Double,Model[_],ParamMap)]()
    var currentResult_ = ListMap[Int, ParamMap]()

    for ( i <-  0 to (s ).toInt)
    {
      println("       -- Number of items:" + currentResult.size )
      if( !IsTimeOut()) {
        //Run each of the n_i configs for r_i iterations and keep best n_i/eta
        var n_i = n * math.pow(eeta, (-i))
        var r_i: Double = r * math.pow(eeta, (i))
        filelog.logException("-- Loop Number " + i + " , check " + n_i + "Hyperparameter values on " + formatter.format(r_i) + "% of the data\n")
        println("       -- Loop Number " + i + " , check " + n_i + "Hyperparameter values on " + formatter.format(r_i) + "% of the data")
        //var resultParam =
          if (i == 0) {
            var tmp = learn(dataset, SelectedParamsMapIndexed, r_i)
            if( tmp.size > 0)
              currentResult = tmp
            else
              println("      -- learn return null at first loop in this sh")

          }
          else {
            for( u <- 0 until n_i.toInt)
              {
                val ind = currentResult.maxBy{ case (key, value) => value._1 }._1
                currentResult_ += (ind -> currentResult(ind)._3)
                currentResult -= ind
              }
            var tmp = learn(dataset, currentResult_, r_i)

           if(  tmp.size > 0)
              currentResult = tmp
            else
              println("       --learn return null at this loop and we have:" + currentResult.size + " items in the list" )
          }
      }
      else
      {
        println("       -- Time out @ SH")
        if(currentResult.size > 0 ) {
          println("       --> best accuracy (after time out) in this sh:" + ListMap(currentResult.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1).toList(0)._2  )

          val ind = currentResult.maxBy{ case (key, value) => value._1 }._1
          val res = currentResult.filterKeys( k => k == ind)
          return res//ListMap(res._3 -> (res._1 , res._2))
          //return ListMap(currentResult.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1)
        }
        else
          null

      }
    }
    // return best hyper parameters for this SH run
    if(currentResult.size > 0 ) {
      println("       -- best accuracy (after end of sh) in this sh:" + ListMap(currentResult.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1).toList(0)._2  )

      val ind = currentResult.maxBy{ case (key, value) => value._1 }._1
      val res = currentResult.filterKeys( k => k == ind)
      return res//ListMap(res._3 -> (res._1 , res._2.asInstanceOf[Model[_]]))

      //return ListMap(currentResult.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1)

    }
    else null

  }

  /*

  * */

  /**
    * this function ML algorithm training for a set of hyper parameter configuration, the training could be parallelized (if the cluster can has free nodes)
    * @param dataset The input dataset on which we will run the hyperband
    * @param param the yperparameters
    * @param r
    * @return
    */
  def learn(dataset: Dataset[_] , param : Map[Int,ParamMap] , r : Double): ListMap[Int , (Double,Model[_],ParamMap)] = {
    try {
      val schema = dataset.schema
      transformSchema(schema, logging = true)
      val est = $(estimator)
      val eval = $(evaluator)
      val epm = param.values.toList
      val shouldLogtoFile = $(logToFile)


      // Create execution context based on $(parallelism)
      val executionContext = getExecutionContext

      val Array(trainingDataset, validationDataset) =
        dataset.randomSplit(Array(0.8, 0.2), $(seed))

      val Array(partation, restofdata) =
        trainingDataset.randomSplit(Array(r / 100, 1 - (r / 100)), $(seed))




      // cache data
      trainingDataset.cache()
      validationDataset.cache()

     //Map to save the result
      var iterResultMap = collection.mutable.Map[Int, (Double,Model[_],ParamMap) ]()


      // Fit models in a Future for training in parallel
      //val metricFutures = epm.zipWithIndex.map { case (paramMap, paramIndex) =>
        val metricFutures = param.map { case (paramIndex ,paramMap ) =>
        Future[Double] {
          if( ! IsTimeOut()) {
            //println(" =====> Remaining time before ML:" + getRemainingTime())
            val model = est.fit(partation, paramMap).asInstanceOf[Model[_]]
            val metric = eval.evaluate(model.transform(validationDataset, paramMap))
            //println(", Accuracy:" + metric)
            //pw.write(" Parameters:" + paramMap.toSeq.toString() + ", Metric:" + metric + "\n")
            //println("-- -- Metric:" + metric)
            //println("paramMap:" + paramMap.toString())

            if ($(ClassifierName) == "RandomForestClassifier")
              iterResultMap += (paramIndex -> (metric, model.asInstanceOf[RandomForestClassificationModel], paramMap))
            else if ($(ClassifierName) == "LogisticRegression")
              iterResultMap += (paramIndex -> (metric, model.asInstanceOf[LogisticRegressionModel], paramMap))
            else if ($(ClassifierName) == "DecisionTreeClassifier")
              iterResultMap += (paramIndex -> (metric, model.asInstanceOf[DecisionTreeClassificationModel], paramMap))
            else if ($(ClassifierName) == "MultilayerPerceptronClassifier")
              iterResultMap += (paramIndex -> (metric, model.asInstanceOf[MultilayerPerceptronClassificationModel], paramMap))
            else if ($(ClassifierName) == "LinearSVC")
              iterResultMap += (paramIndex -> (metric, model.asInstanceOf[LinearSVCModel], paramMap))
            else if ($(ClassifierName) == "NaiveBayes")
              iterResultMap += (paramIndex -> (metric, model.asInstanceOf[NaiveBayesModel], paramMap))
            else if ($(ClassifierName) == "GBTClassifier")
              iterResultMap += (paramIndex -> (metric, model.asInstanceOf[GBTClassificationModel], paramMap))
            else if ($(ClassifierName) == "LDA")
              iterResultMap += (paramIndex -> (metric, model.asInstanceOf[LDAModel], paramMap))
            else
              iterResultMap += (paramIndex -> (metric, model.asInstanceOf[QDAModel], paramMap))

            //println("     - Accuracy:" + metric )
            //filelog.logException("Accuracy:" + metric.toString + " ,Parameters:" + paramMap.toSeq.mkString + "\n")
            metric
          }else {
            //println("===>>>Time out in future")
            0.0
          }

        }(executionContext)
      }


      import scala.concurrent.duration._
      val duration = Duration(getRemainingTime(), MILLISECONDS)

      // Wait for all metrics to be calculated
      try {
        //println(" =====> Remaining time before 2 ML:" + getRemainingTime())
        val metrics = metricFutures.map(ThreadUtils.awaitResult(_, duration)) //Duration.Inf))
        //println(" =====> Remaining time after ML:" + getRemainingTime())
      }catch
       {
          case ex:Exception => println("      --TimeOut:==>" +ex.getMessage)
           println(ex.getStackTrace())
       }

      //println(">>>>> Count found" + iterResultMap.size)
      var sortedIterResultMap =
        if (eval.isLargerBetter)
          ListMap(iterResultMap.toSeq.sortWith( (x1,x2) => x1._2._1.toString + x1._1.toString > x2._2._1.toString + x2._1.toString): _*)
          //ListMap(iterResultMap.toSeq.sortWith( (x1,x2) => x1._2._1 + x1._1.toSeq(2).value.toString + x1._1.toSeq(1).value.toString + x1._1.toSeq(0).value.toString >
           //                                        x2._2._1 + x2._1.toSeq(2).value.toString + x2._1.toSeq(1).value.toString + x2._1.toSeq(0).value.toString): _*)
          //ListMap(iterResultMap.toSeq.sortWith(_._2._1 > _._2._1): _*)
        else
          ListMap(iterResultMap.toSeq.sortWith(_._2._1 < _._2._1): _*)
      // Unpersist training & validation set once all metrics have been produced
      trainingDataset.unpersist()
      validationDataset.unpersist()


      //println("     ------ best is " + sortedIterResultMap.head._2 + "-----")
      sortedIterResultMap
   }catch
      {
        case ex:Exception =>
          println("Exception (Hyperband - "+$(ClassifierName)+"- learn): " + ex.getMessage)
          ex.printStackTrace()
          return null
      }
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






