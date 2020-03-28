package org.apache.spark.ml.tuning


import java.text.DecimalFormat
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
    * data percentage starting point
    * added to r in each sh iteration
    * Default: 0
    *
    * @group param
    */
  val basicDataPercentage: DoubleParam = new DoubleParam(this, "basicDataPerventage",
    "tata percentage starting point")

  /** @group getParam */
  def getbasicDataPerventage: Double = $(basicDataPercentage)

  setDefault(basicDataPercentage -> 0.0)



  /**
    * skipped SH session in hyperband
    * Default: 0, skip no session
    *
    * @group param
    */
  val skipSH: IntParam = new IntParam(this, "skipSH",
    "number of session to be skiped in hyperband (if any)", ParamValidators.inRange(0, 2))

  /** @group getParam */
  def getskipSH: Int = $(skipSH)

  setDefault(skipSH -> 0)


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
    * should i do random split per class
    *
    * @group param
    */
  val SplitbyClass: BooleanParam = new BooleanParam(this, "SplitbyClass"," should i do random split per class")

  /** @group getParam */
  def getSplitbyClass: Boolean = $(SplitbyClass)

  setDefault(SplitbyClass -> false)



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
  * Parameters for the optimization process result
  */
trait OptimizerResult
{
  var bestParam :ParamMap = null
  var bestModel :Model[_] = null
  var bestmetric :Double = 0.0
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
    with OptimizerResult
    with MLWritable with Logging {

  val fm2d = new DecimalFormat("###.##")
  val fm4d = new DecimalFormat("###.####")
  var filelog :Logger = null
  var ClassifierParamsMapIndexed =  Map[Int,ParamMap]()
  var ClassifiersMgr: ClassifiersManager = null
  var StartingTime:Date = null
  val formatter = java.text.NumberFormat.getInstance
  formatter.setMaximumFractionDigits(2)

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

  @Since("2.3.0")
  def setClassifierName(value: String): this.type = set(ClassifierName, value)

  @Since("2.3.0")
  def setmaxTime(value: Long): this.type = set(maxTime, value)

  @Since("2.3.0")
  def setskipSH(value: Int): this.type = set(skipSH, value)

  def setSplitbyClass(value:Boolean) : this.type = set(SplitbyClass,value)

  def setbasicDataPercentage(value:Double) : this.type = set(basicDataPercentage , value)

  /**
    * check if timeout or not
    * @return
    */
  def IsTimeOut(): Boolean =  {
    if ( getRemainingTime() == 0)
      return true
    else
      return false
  }

  /**
    * get remaining time
    * @return
    */
  def getRemainingTime(): Long =  {
    var rem:Long = ($(maxTime) * 1000) - (new Date().getTime - StartingTime.getTime())
    if (rem < 0)
      rem = 0
    return rem
  }

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
      val res = hyberband(dataset)

      if(res.size > 0) {
        // get the best parameters returned by the Hyperband
        var bestParam: ParamMap = ListMap(res.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1).keys.head

        // train model using best parameters
        val bestModel: Model[_] = ListMap(res.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1).values.head._2

        // evaluate the best Model
        val metric: Double = ListMap(res.toSeq.sortWith(_._2._1 > _._2._1): _*).take(1).values.head._1

        this.bestParam = bestParam
        this.bestModel = bestModel
        this.bestmetric = metric

        // return Hyperband mode (with: best model, best parameters and its evaluation metric)
        return new HyperbandModel(uid, bestModel, Array(metric))
      }
      else
        return null
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
 // properities of hyperband
    val eeta = $(eta)
    val max_Resource = $(maxResource)
    val shouldLogtoFile = $(logToFile)

    // List of parameters for each sh iteration
    var currentResult = ListMap[ParamMap, (Double,Model[_])]()

    // Log eta
    var logeta =  (x:Double) =>  Math.log(x)/Math.log(eeta)

    // number of unique executions of Successive Halving (minus one)
    var s_max = math.round(logeta(max_Resource))

    //Budget (without reuse) per execution of Succesive Halving (n,r)
    var B = (s_max+1) * max_Resource

    var firstSH = true
    // loop (number of successive halving, with different number of hyper-parameters configurations)
    // incearsing number of configuration mean decreasing the resource per each configuration
    var time_out = false
    // s_max- skipSH to remove the first n loop from hyperband session

    for( s <- ( s_max-$(skipSH) ) to 0 by -1) {
      if (!IsTimeOut()) {

        //initial number of configurations
        var tmp = math.ceil((B / max_Resource / (s + 1)))
        var n = math.round(tmp * math.pow(eeta, s))

        //initial number of resources to run configurations for
        var r = max_Resource * math.pow(eeta, (-s))
        println("     -- Successive Halving Session:" + s + " , Starting with " + formatter.format(r) + "% of data and "  + n +" Models to train")
        filelog.logOutput("     -- Successive Halving Session:" + s + " , Starting with " + formatter.format(r) + "% of data and "  + n +" Models to train\n")

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
      else {
          //currentResult.foreach( e => println(e._2._1))
        if(!time_out) {
          filelog.logOutput("     --Time out @ Main Hyperband loop\n")
          time_out = true
        }
          return currentResult
        }
    }

    currentResult
  }

  /**
    * Get best Accuracy in a Map List
    * @param lst
    * @return
    */
  def getBestAcc( lst:Map[Int, (Double,Model[_],ParamMap)]):(Int, (Double,Model[_],ParamMap)) = {
    var curracc = 0.0
    var currind = 1
    var currentResult = ListMap[Int, (Double,Model[_],ParamMap)]()
    for( i <- lst.keys)
      {
        if( lst(i)._1 > curracc )
          {
            curracc = lst(i)._1
            currind = i
          }
      }
    return ( currind , lst(currind) )
  }


  /*
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

    var SelectedParamsMapIndexed =  Map[Int,ParamMap]()
    SelectedParamsMapIndexed = ClassifiersMgr.getNRandomParameters( $(ClassifierName), n , s)//ClassifiersMgr.getRandomParametersIndexed( $(ClassifierName), n)

    // for debuging convert ParamMap to string
    var s12 = ""
    for ( ps <- SelectedParamsMapIndexed) {
      for (p <- ps._2.toSeq) {
        s12 = p.param.name + ":" + p.value + "," + s12

      }
      s12 = ""
    }

    // list to save hyper parameter configuration during the sh iteration
    var currentResult = ListMap[Int, (Double,Model[_],ParamMap)]()
    var BestResult = ListMap[Int, (Double,Model[_],ParamMap)]()
    var currentResult_ = ListMap[Int, ParamMap]()

    var counter = 0
    var time_out = false
    for ( i <-  0 to (s ).toInt)
    {
      // if no time out
      if( !IsTimeOut()) {
        //Run each of the n_i configs for r_i iterations and keep best n_i/eta (loop)
        var n_i = n * math.pow(eeta, (-i))
        var r_i: Double = r * math.pow(eeta, (i))
        print("       -- Loop Number " + i + " , Train " + n_i + " Models on " + formatter.format(r_i) + "% of the data")
        filelog.logOutput("       -- Loop Number " + i + " , Train " + n_i + " Models on " + formatter.format(r_i) + "% of the data")

        //if this is the first loop
        if (i == 0) {
            var tmp = learn(dataset, SelectedParamsMapIndexed, r_i)
            if( tmp != null) {
              counter = counter + 1
              BestResult += ( counter -> getBestAcc(tmp)._2)
              currentResult = tmp
            }
            else {
              //println("      -- learn return null at first loop in this sh")
              //filelog.logOutput("      -- learn return null at first loop in this sh\n")
            }
          }
          else {
          // not the first loop (loop to get best param)
          currentResult_ = currentResult_.empty
          for( u <- 0 until n_i.toInt)
              {
                val ind = currentResult.maxBy{ case (key, value) => value._1 }._1
                currentResult_ += (ind -> currentResult(ind)._3)
                currentResult -= ind
              }
            //println("Number of parm:" +currentResult_.size)
            var tmp = learn(dataset, currentResult_, r_i)

           if(  tmp != null) {
             currentResult = tmp
             counter = counter + 1
             BestResult += ( counter -> getBestAcc(tmp)._2)
           }
            else {
              //println("       --learn return null at this loop and we have:" + currentResult.size + " items in the list")
              //filelog.logOutput("       --learn return null at this loop and we have:" + currentResult.size + " items in the list\n")
           }
           }
      }
      // time out
      else {
        if(!time_out) {
          println("       -- Time out @ SH")
          filelog.logOutput("       -- Time out @ SH\n")
          time_out = true
        }
        if(currentResult.size > 0 ) {
          val ind = BestResult.maxBy{ case (key, value) => value._1 }._1
          val res = BestResult.filterKeys( k => k == ind)
          println("       --> best accuracy (after time out) in this sh:" + fm4d.format( 100 * res(ind)._1 ) + "%")
          filelog.logOutput("       --> best accuracy (after time out) in this sh:" + fm4d.format(100 * res(ind)._1 ) + "%\n" )
          return res
        }
        else
          null
      }
    }
    // return best hyper parameters for this SH run
    if(currentResult.size > 0 ) {
      val ind = BestResult.maxBy{ case (key, value) => value._1 }._1
      val res = BestResult.filterKeys( k => k == ind)
      println("       -->> best accuracy for this session:" + fm4d.format( 100 * res(ind)._1 ) + "%")
      filelog.logOutput("       -->> best accuracy for this session:" + fm4d.format( 100 * res(ind)._1 ) + "%\n" )
      return res

    }
    else null

  }


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
      val epm = ListMap(param.toSeq.sortBy(_._1):_*).values.toList
      val shouldLogtoFile = $(logToFile)
      val starttime1 = new java.util.Date().getTime

      // Create execution context based on $(parallelism)
      val executionContext = getExecutionContext

      val Array(trainingDataset, validationDataset) =
        dataset.randomSplit(Array(0.8, 0.2), $(seed))

      //val Array(partation, restofdata) =  trainingDataset.randomSplit(Array(r / 100, 1 - (r / 100)), $(seed))

      val partation =  RandomSplitByClassValues( trainingDataset , r / 100 , $(SplitbyClass))

      // cache data
      trainingDataset.cache()
      validationDataset.cache()

     //Map to save the result
      var iterResultMap = collection.mutable.Map[Int, (Double,Model[_],ParamMap) ]()


       // Fit models in a Future for training in parallel
       val metricFutures = param.map { case ( paramIndex , paramMap) =>
          Future[Double] {
            if (!IsTimeOut()) {
              //val paramIndex:Int = 1
              val model = est.fit(partation, paramMap).asInstanceOf[Model[_]]
              val metric = eval.evaluate(model.transform(validationDataset, paramMap))

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

              metric
            } else {
              0.0
            }

          }(executionContext)
        }
        import scala.concurrent.duration._
        val duration = Duration(getRemainingTime(), MILLISECONDS)

        // Wait for all metrics to be calculated
        try {
          //println(" (remaining Time1:" + getRemainingTime() + ")")
          val metrics = metricFutures.map(ThreadUtils.awaitResult(_, duration))
          //println("    -- iteration" + Index)
        } catch {

          case ex: Exception => //println("(remaining Time2:" + getRemainingTime() +")")
            println("      -->(TimeOut...)") //+ex.getMessage)
        }

      //}
       var sortedIterResultMap =
       if(iterResultMap.size > 0 ) {
         if (eval.isLargerBetter)
           ListMap(iterResultMap.toSeq.sortWith((x1, x2) => x1._2._1.toString + x1._1.toString > x2._2._1.toString + x2._1.toString): _*)
         else
           ListMap(iterResultMap.toSeq.sortWith(_._2._1 < _._2._1): _*)
       } else
       null

        // Unpersist training & validation set once all metrics have been produced
        trainingDataset.unpersist()
        validationDataset.unpersist()

        val Endtime1 = new java.util.Date().getTime
        val TotalTime1 = Endtime1 - starttime1
        print(" , Time(" + (TotalTime1 / 1000.0).toString + ") ")
        filelog.logOutput(" , Time(" + (TotalTime1 / 1000.0).toString + ") ")

      if( sortedIterResultMap != null ) {
        println(" and Best Accuracy is: " + fm4d.format(100 * sortedIterResultMap.head._2._1) + "%")
        filelog.logOutput(" and Best Accuracy is: " + fm4d.format(100 * sortedIterResultMap.head._2._1) + "%\n")
      }
        sortedIterResultMap

   }catch
      {
        case ex:Exception =>
          println("Exception (Hyperband - "+$(ClassifierName)+"- learn): " + ex.getMessage)
          return null
      }
  }

  /**
    * Split Data, this function can do splitting randomly.
    * it has option that allow us to split data based taking classes ration into consideration
    * @param dataset
    * @param Percentage
    * @param SplitbyClass
    * @return
    */
  def RandomSplitByClassValues(dataset: Dataset[_] , Percentage: Double , SplitbyClass:Boolean = false): Dataset[_] =  {
    var bdf: DataFrame = null

    var Percentage_updated = Percentage + $(basicDataPercentage)
    if (Percentage_updated > 1.0)
      Percentage_updated = 1.0

    val StartTime = new java.util.Date().getTime
     if (Percentage_updated > 0.75 || !SplitbyClass)
      {
        val Array(result, restofdata) =  dataset.randomSplit(Array(Percentage_updated, 1 - Percentage_updated), $(seed))
        bdf = result.toDF()
      }
      else
      {
        var labelValues = dataset.select("y").distinct.collect.flatMap(_.toSeq)
        var byLabelValuesArray = labelValues.map(lv => dataset.toDF().where("y ==" + lv))

        for (ii <- 0 to labelValues.size - 1) {
          //println("loop:" + ii)
          var Array(result, unwanted) = byLabelValuesArray(ii).randomSplit(Array(Percentage_updated, 1 - Percentage_updated), $(seed))
          if (ii == 0) {
            bdf = result
            //println("Iter:" + ii + "  , Count:" + bdf.count)
          }
          else {
            bdf = bdf.union(result)
            //println("Iter:" + ii + "  , Count:" + bdf.count)
          }
        }
      }

    val Endtime = new java.util.Date().getTime
    val TotalTime = Endtime - StartTime
    //println("      ->split for: "+ Percentage + ", time:" + TotalTime/1000.0)
    return bdf
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
class HyperbandModel  (
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






