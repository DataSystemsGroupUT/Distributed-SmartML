package org.apache.spark.ml.tuning

import java.text.DecimalFormat
import java.util.{Date, Locale, List => JList}

import org.apache.hadoop.fs.Path
import org.apache.spark.annotation.Since
import org.apache.spark.internal.Logging
import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.{Evaluator, RegressionEvaluator}
import org.apache.spark.ml.feature.StringIndexer
import org.apache.spark.ml.param._
import org.apache.spark.ml.param.shared.{HasCollectSubModels, HasParallelism}
import org.apache.spark.ml.regression.{RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.util._
import org.apache.spark.ml.{Estimator, Model}
import org.apache.spark.sql.functions.desc
import org.apache.spark.sql.types.StructType
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.util.ThreadUtils
import org.dsmartml._
import org.json4s.DefaultFormats

import scala.collection.JavaConverters._
import scala.concurrent.Future
import scala.language.existentials
import scala.collection.immutable.ListMap
import scala.collection.mutable.ListBuffer
import scala.util.control.Breaks.{break, breakable}


/**
  * Parameters for [[BayesianOpt]] and [[BayesianOptModel]].
  */
trait BayesianOptParams extends ValidatorParams {
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
    * the maximum Time allowed for BayesianOpt on this algorithm
    * Default: Inf
    *
    * @group param
    */
  val maxTime: LongParam = new LongParam(this, "maxTime",
    "the maximum amount of Time (in seconds) that can\nbe allocated to BayesianOpt algorithm")
  /** @group getParam */
  def getmaxTime: Double = $(maxTime)
  setDefault(maxTime -> 60 * 10) // 10 minutes


  val paramSeed: IntParam = new IntParam(this, "paramSeed",
    "the seed number of random parameters", ParamValidators.inRange(1, 10000))
  /** @group getParam */
  def getparamSeed: Int = $(paramSeed)
  setDefault(paramSeed -> 1234)


  /**
    * controls the Number of
    * configurations of hyperparameters initially tested
    * Default: 5
    * @group param
    */
  val initialParamNumber: IntParam = new IntParam(this, "InitialParamNumber",
    "the number of random parameters initially tried", ParamValidators.inRange(1, 10000))
  /** @group getParam */
  def getinitialParamNumber: Int = $(initialParamNumber)
  setDefault(initialParamNumber -> 5)

  /**
    * controls the Max Number of
    * Bayesian Steps tested
    * Default: 20
    * @group param
    */
  val bayesianStepsNumber: IntParam = new IntParam(this, "BayesianStepsNumber",
    "the number of Bayesian Steps  tried", ParamValidators.inRange(1, 10000))
  /** @group getParam */
  def getbayesianStepsNumber: Int = $(bayesianStepsNumber)
  setDefault(bayesianStepsNumber -> 20)

  /**
    * data percentage starting point
    * Default: 10%
    *
    * @group param
    */
  val initialDataPercentage: DoubleParam = new DoubleParam(this, "initialDataPercentage",
    "Initial Data percentage used with the initial random configuration points")
  /** @group getParam */
  def getinitialDataPercentage: Double = $(initialDataPercentage)
  setDefault(initialDataPercentage -> 0.1)

  /**
    * data percentage steps for bayesian point
    * Default: 5%
    *
    * @group param
    */
  val bayesianStepDataPercentage: DoubleParam = new DoubleParam(this, "bayesianStepDataPercentage",
    "Data percentage added to each bayesian step")
  /** @group getParam */
  def getbayesianStepDataPercentage: Double = $(bayesianStepDataPercentage)
  setDefault(bayesianStepDataPercentage -> 0.5)

  /**
    * Target Column name
    *
    * @group param
    */
  val TargetColumn: Param[String] = new Param[String](this, "TargetColumn","Target Column name")
  /** @group getParam */
  def getTargetColumn: String = $(TargetColumn)
  setDefault(TargetColumn -> "y")

}


/**
  * Description: this calss represent Bayesian Opt (using Random Fores Regression)
  * implementation & it support parallelism & some sort of Hyperband by allocating
  * more resource (train on more data) for each step
  * @constructor Create a Bayesian Opt Object all us to do Bayesian optimization
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 10/3/2020
  */
class BayesianOpt (@Since("1.5.0") override val uid: String)
  extends Estimator[BayesianOptModel]
    with BayesianOptParams with HasParallelism with HasCollectSubModels
    with OptimizerResult
    with MLWritable with Logging{


  val fm2d = new DecimalFormat("###.##")
  val fm4d = new DecimalFormat("###.####")
  var filelog :Logger = null
  var ClassifierParamsMapIndexed =  Map[Int,ParamMap]()
  var ClassifiersMgr: ClassifiersManager = null
  var spark:SparkSession = null
  var StartingTime:Date = null
  var Allparam_df:DataFrame = null
  var paramMapList = new ListBuffer[ParamMap]()
  var ParamMapAccuracyLst = new ListBuffer[(ParamMap, Double , Model[_])]()
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
  def setTargetColumn(value: String): this.type = set(TargetColumn, value)

  @Since("2.3.0")
  def setinitialParamNumber(value: Int): this.type = set(initialParamNumber, value)
  @Since("2.3.0")
  def setbayesianStepsNumber(value: Int): this.type = set(bayesianStepsNumber, value)
  @Since("2.3.0")
  def setinitialDataPercentage(value: Double): this.type = set(initialDataPercentage, value)
  @Since("2.3.0")
  def setbayesianStepDataPercentage(value: Double): this.type = set(bayesianStepDataPercentage, value)

  @Since("2.3.0")
  def setparamSeed(value: Int): this.type = set(paramSeed, value)


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
    * This function Run BayesianOpt Algorithm on the data set to get best algorithm (best hyper parameters)
    * @param dataset the input dataset on which we will run the BayesianOpt
    * @return BayesianOptModel object (represent best model found based on accuracy and its hyperparameters)
    */
  @Since("2.0.0")
  override def fit(dataset: Dataset[_]): BayesianOptModel = {
    if (!IsTimeOut()) {
      // properities of BayesianOpt
      val est = $(estimator)
      val eval = $(evaluator)
      val logpath = $(logFilePath)
      val shouldLogToFile = $(logToFile)

      val schema = dataset.schema
      transformSchema(schema, logging = true)


      // get All Random Parameters for this classifier
      var AllparamMapList = ClassifiersMgr.generateParametersForBayesianOpt($(ClassifierName) ) //"RandomForestClassifier")

      // Convert all the parameters to List of (ParamMap, Accuracy, Model)
      var AllParamMapAccLst = new ListBuffer[ (ParamMap , Double , Model[_])]()
      AllparamMapList.foreach( e => AllParamMapAccLst.append( (e,1,null) ))

      // Convert All Parameters List to Dataframe
      Allparam_df = convertParamtoDF(spark, AllParamMapAccLst , true, $(ClassifierName))

      // Convert the parameters Dataframe to Vector Assembly format (without label, only parametes)
      Allparam_df = DataLoader.convertDFtoVecAssembly_WithoutLabel( Allparam_df , "features" )

      // select n initial Configuration (randomly)
      //-------------------------------------------------------------------------------
      val count = AllparamMapList.length;
      import scala.util.Random
      val rand = new Random($(paramSeed))
      val numberOfElements = $(initialParamNumber)
      var i = 0
      while ( i < numberOfElements)
      {
        val randomIndex = rand.nextInt(AllparamMapList.size)
        //println(randomIndex + "...")
        val randomElement = AllparamMapList(randomIndex)
        paramMapList.append(randomElement)
        i += 1
      }

      // do initial training n times for n random hyperparameters
      //  then loop to do sequentail training for the selected parameters
      //----------------------------------------------------------------------------
      if (!IsTimeOut()) {
        BayeisanInitialStep(dataset.toDF(), $(ClassifierName), ClassifiersMgr)
        for (i <- 1 to $(bayesianStepsNumber)) {
          if (!IsTimeOut()) {
            BayeisanStep(spark, dataset.toDF(), ClassifiersMgr, i, $(ClassifierName))
          }
        }
      }


      // Get best Hyperparameters, Model and Accuracy and set the class properities
      // also return Bayesian Model
      //----------------------------------------------------------------------------
      if(ParamMapAccuracyLst.size > 0) {
        // get the best parameters returned by the BayesianOpt
        var bestParam: ParamMap = ParamMapAccuracyLst.sortWith(_._2 > _._2)(0)._1

        // train model using best parameters
        val bestModel: Model[_] = ParamMapAccuracyLst.sortWith(_._2 > _._2)(0)._3

        // evaluate the best Model
        val metric: Double = ParamMapAccuracyLst.sortWith(_._2 > _._2)(0)._2.toDouble

        this.bestParam = bestParam
        this.bestModel = bestModel
        this.bestmetric = metric

        // return BayesianOpt mode (with: best model, best parameters and its evaluation metric)
        return new BayesianOptModel(uid, bestModel, Array(metric))
      }
      else
        return null
    }
    else
      return null
  }

  override def transformSchema(schema: StructType): StructType = transformSchemaImpl(schema)

  @Since("1.5.0")
  override def copy(extra: ParamMap): BayesianOpt = {
    val copied = defaultCopy(extra).asInstanceOf[BayesianOpt]
    if (copied.isDefined(estimator)) {
      copied.setEstimator(copied.getEstimator.copy(extra))
    }
    if (copied.isDefined(evaluator)) {
      copied.setEvaluator(copied.getEvaluator.copy(extra))
    }
    copied
  }

  @Since("2.0.0")
  override def write: MLWriter = new BayesianOpt.BayesianOptWriter(this)

  /**
    * This function convert List of ParamMap to Dataframe with column for each param
    * it can also add a label column or not based on the parameter "WithoutLable"
    * this parameter represent the accuracy of the hyperparameters
    * @param spark
    * @param paramMapList
    * @param WithoutLabel
    * @param Classifier
    * @return
    */
  def convertParamtoDF(spark:SparkSession,paramMapList:ListBuffer[(ParamMap, Double, Model[_])], WithoutLabel:Boolean , Classifier:String): DataFrame =
  {
    import spark.implicits._
    if(Classifier == "RandomForestClassifier") {
      var RF_Param_Lst = new ListBuffer[(Int, Int, String, Int, Int, Double, Double)]()
      for (pm <- paramMapList) {
        var maxBins = 0
        var minInstancesPerNode = 0
        var impurity = ""
        var numTrees = 0
        var maxDepth = 0
        var minInfoGain = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxBins")
            maxBins = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInstancesPerNode")
            minInstancesPerNode = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "impurity")
            impurity = parmmap.value.asInstanceOf[String]
          if (parmmap.param.name == "numTrees")
            numTrees = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "maxDepth")
            maxDepth = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInfoGain")
            minInfoGain = parmmap.value.asInstanceOf[Double]
        }
        RF_Param_Lst.append((maxBins, minInstancesPerNode, impurity, numTrees, maxDepth, minInfoGain, pm._2))
      }
      var columns = Seq("maxBins", "minInstancesPerNode", "impurity", "numTrees", "maxDepth", "minInfoGain" , "Label")
      val data = RF_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("maxBins", "minInstancesPerNode", "impurity", "numTrees", "maxDepth", "minInfoGain" , "Label")
      val indexer = new StringIndexer().setInputCol("impurity").setOutputCol("c_impurity").fit(df)
      df = indexer.transform(df).drop("impurity")
      if(WithoutLabel)
        df.drop("Label")
      return df
    }
    if(Classifier == "LogisticRegression"){
      var LR_Param_Lst = new ListBuffer[(Boolean, Int, Double, Double, Boolean, Double, Double)]()
      //fitIntercept (Boolean) - maxIter (Int) - regParam (Double)- elasticNetParam (Double)- standardization (bool)- tol (Double)
      for (pm <- paramMapList) {
        var fitIntercept = false
        var maxIter = 0
        var regParam = 0.0
        var elasticNetParam = 0.0
        var standardization = false
        var tol = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "fitIntercept")
            fitIntercept = parmmap.value.asInstanceOf[Boolean]
          if (parmmap.param.name == "maxIter")
            maxIter = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "regParam")
            regParam = parmmap.value.asInstanceOf[Double]
          if (parmmap.param.name == "elasticNetParam")
            elasticNetParam = parmmap.value.asInstanceOf[Double]
          if (parmmap.param.name == "standardization")
            standardization = parmmap.value.asInstanceOf[Boolean]
          if (parmmap.param.name == "tol")
            tol = parmmap.value.asInstanceOf[Double]
        }
        LR_Param_Lst.append((fitIntercept,maxIter,regParam,elasticNetParam,standardization,tol,pm._2))
      }
      val data = LR_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("fitIntercept", "maxIter", "regParam", "elasticNetParam", "standardization", "tol" , "Label")

      //      val indexer1 = new StringIndexer().setInputCol("fitIntercept").setOutputCol("c_fitIntercept").fit(df)
      //      df = indexer1.transform(df).drop("fitIntercept")
      //      val indexer2 = new StringIndexer().setInputCol("standardization").setOutputCol("c_standardization").fit(df)
      //      df = indexer2.transform(df).drop("standardization")
      if(WithoutLabel)
        df.drop("Label")
      return df

    }
    if(Classifier == "DecisionTreeClassifier") {
      var RF_Param_Lst = new ListBuffer[(Int, Int, String, Int, Double, Double)]()
      for (pm <- paramMapList) {
        var maxBins = 0
        var minInstancesPerNode = 0
        var impurity = ""
        var maxDepth = 0
        var minInfoGain = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxBins")
            maxBins = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInstancesPerNode")
            minInstancesPerNode = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "impurity")
            impurity = parmmap.value.asInstanceOf[String]
          if (parmmap.param.name == "maxDepth")
            maxDepth = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInfoGain")
            minInfoGain = parmmap.value.asInstanceOf[Double]
        }
        RF_Param_Lst.append((maxBins, minInstancesPerNode, impurity, maxDepth, minInfoGain, pm._2))
      }
      var columns = Seq("maxBins", "minInstancesPerNode", "impurity",  "maxDepth", "minInfoGain" , "Label")
      val data = RF_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("maxBins", "minInstancesPerNode", "impurity",  "maxDepth", "minInfoGain" , "Label")
      val indexer = new StringIndexer().setInputCol("impurity").setOutputCol("c_impurity").fit(df)
      df = indexer.transform(df).drop("impurity")
      if(WithoutLabel)
        df.drop("Label")
      return df
    }
    if(Classifier == "GBTClassifier") {
      var RF_Param_Lst = new ListBuffer[(Int, Int, String, Int, Double, Double)]()
      for (pm <- paramMapList) {
        var maxBins = 0
        var minInstancesPerNode = 0
        var impurity = ""
        var maxDepth = 0
        var minInfoGain = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxBins")
            maxBins = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInstancesPerNode")
            minInstancesPerNode = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "impurity")
            impurity = parmmap.value.asInstanceOf[String]
          if (parmmap.param.name == "maxDepth")
            maxDepth = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInfoGain")
            minInfoGain = parmmap.value.asInstanceOf[Double]
        }
        RF_Param_Lst.append((maxBins, minInstancesPerNode, impurity, maxDepth, minInfoGain, pm._2))
      }
      var columns = Seq("maxBins", "minInstancesPerNode", "impurity",  "maxDepth", "minInfoGain" , "Label")
      val data = RF_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("maxBins", "minInstancesPerNode", "impurity",  "maxDepth", "minInfoGain" , "Label")
      val indexer = new StringIndexer().setInputCol("impurity").setOutputCol("c_impurity").fit(df)
      df = indexer.transform(df).drop("impurity")
      if(WithoutLabel)
        df.drop("Label")
      return df
    }
    if(Classifier == "LinearSVC"){
      var LR_Param_Lst = new ListBuffer[(Int, Double, Double)]()
      //fitIntercept (Boolean) - maxIter (Int) - regParam (Double)- elasticNetParam (Double)- standardization (bool)- tol (Double)
      for (pm <- paramMapList) {
        var maxIter = 0
        var regParam = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxIter")
            maxIter = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "regParam")
            regParam = parmmap.value.asInstanceOf[Double]
        }
        LR_Param_Lst.append((maxIter,regParam,pm._2))
      }
      val data = LR_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("maxIter", "regParam",  "Label")

      //      val indexer1 = new StringIndexer().setInputCol("fitIntercept").setOutputCol("c_fitIntercept").fit(df)
      //      df = indexer1.transform(df).drop("fitIntercept")
      //      val indexer2 = new StringIndexer().setInputCol("standardization").setOutputCol("c_standardization").fit(df)
      //      df = indexer2.transform(df).drop("standardization")
      if(WithoutLabel)
        df.drop("Label")
      return df

    }
    if(Classifier == "MultilayerPerceptronClassifier"){
      var P_Param_Lst = new ListBuffer[(Int, Int,Int,Int, Double)]()
      var layers_0 = 0
      var layers_1 = 0
      var layers_2 = 0
      var maxIter = 0
      for (pm <- paramMapList) {
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxIter")
            maxIter = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "layers") {
            layers_0 = parmmap.value.asInstanceOf[Array[Int]](0)
            layers_1 = parmmap.value.asInstanceOf[Array[Int]](1)
            layers_2 = parmmap.value.asInstanceOf[Array[Int]](2)
          }
        }
        P_Param_Lst.append((maxIter,layers_0,layers_1,layers_2,pm._2))
      }
      val data = P_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF( "maxIter","layers_0" ,"layers_1" ,"layers_2" ,  "Label")
      if(WithoutLabel)
        df.drop("Label")
      return df

    }
    return null

  }

  /**
    * this function represent the initial step, it train n models based on n number of
    * hyperparamters config selected randomly.
    * this function call Learn function to do parallel training for the selected algorithm
    * @param df
    * @param classifierName
    * @param classifiermgr
    */
  def BayeisanInitialStep(df:DataFrame , classifierName:String,  classifiermgr:ClassifiersManager): Unit =
  {
    learn(df, paramMapList.toList, classifierName , classifiermgr)
  }

  /**
    * this function train an algorithm for each Parameter Map in the parameter map list
    * it do that on a fraction of the data (initialDataPercentage)
    * it return list item for each training run (PAramMap, Accuracy and the model itself)
    * @param dataset
    * @param paramLst
    * @param ClassifierName
    * @param ClassifierMgr
    * @return
    */
  def learn(dataset: Dataset[_], paramLst: List[ParamMap] , ClassifierName:String , ClassifierMgr:ClassifiersManager): ListBuffer[(ParamMap, Double, Model[_])] =
  {
    try{
    var datapercent = $(initialDataPercentage)
    var result = new ListBuffer[(ParamMap, Double, Model[_])]()
    val Array(trainingDataset, validationDataset) = dataset.randomSplit(Array(datapercent, 0.2), 1234)
    var rf = ClassifierMgr.ClassifiersMap(ClassifierName)

    val est = $(estimator)
    val eval = $(evaluator)

    val executionContext = getExecutionContext
    val metricFutures = paramLst.map { case paramMap =>
      Future[Double] {
        if (!IsTimeOut()) {
          //val paramIndex:Int = 1
          val model = est.fit(trainingDataset, paramMap).asInstanceOf[Model[_]]
          val metric = eval.evaluate(model.transform(validationDataset, paramMap))
          println("      - Initial Random Search (on " + formatter.format(datapercent * 100) + "% of the Data),  Acc:" + formatter.format(metric * 100) + "%")

          if (ClassifierName == "RandomForestClassifier")
            ParamMapAccuracyLst.append((paramMap,metric, model.asInstanceOf[RandomForestClassificationModel]))
          else if (ClassifierName == "LogisticRegression")
            ParamMapAccuracyLst.append((paramMap,metric, model.asInstanceOf[LogisticRegressionModel]))
          else if (ClassifierName == "DecisionTreeClassifier")
            ParamMapAccuracyLst.append((paramMap,metric, model.asInstanceOf[DecisionTreeClassificationModel]))
          else if (ClassifierName == "MultilayerPerceptronClassifier")
            ParamMapAccuracyLst.append((paramMap,metric, model.asInstanceOf[MultilayerPerceptronClassificationModel]))
          else if (ClassifierName == "LinearSVC")
            ParamMapAccuracyLst.append((paramMap,metric, model.asInstanceOf[LinearSVCModel]))
          else if (ClassifierName == "NaiveBayes")
            ParamMapAccuracyLst.append((paramMap,metric, model.asInstanceOf[NaiveBayesModel]))
          else if (ClassifierName == "GBTClassifier")
            ParamMapAccuracyLst.append((paramMap,metric, model.asInstanceOf[GBTClassificationModel]))
          else if (ClassifierName == "LDA")
            ParamMapAccuracyLst.append((paramMap,metric, model.asInstanceOf[LDAModel]))
          else
            ParamMapAccuracyLst.append((paramMap,metric, model.asInstanceOf[QDAModel]))

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


    } catch {
      case ex: Exception =>
        println("Exception (Hyperband - " + ClassifierName + "- learn): " + ex.getMessage)
        ex.printStackTrace()
        return null
    }
    return ParamMapAccuracyLst

  }

  /**
    * helper function to allow us to print Parameter Map
    * @param pm
    * @return
    */
  def printParamMap( pm:ParamMap): String =
  {
    var result = ""
    for ( p <- pm.toSeq)
    {
      result = result +  p.param.name + ":" + p.value + " - "
    }
    //print(result)
    return result
  }

  /**
    * this function do the following
    * 1- get all the tested Hyperparameters + their accuracy and convert them to DF
    * 2- convert this Dataframe to Vector Assembly Format
    * 3- train a regression model on that Datafram
    * 4- use the regression model to select the next paramter    *
    * @param spark
    * @param df_data
    * @param classifiermgr
    * @param step
    * @param classifier
    */
  def BayeisanStep(spark:SparkSession , df_data:DataFrame, classifiermgr:ClassifiersManager , step:Int, classifier:String = "RandomForestClassifier"): Unit =
  {
    //convert result to DF
    var rdf = convertParamtoDF(spark,ParamMapAccuracyLst , false,classifier)

    // convert Result to Assembler
    var bdf = DataLoader.convertDFtoVecAssembly( rdf , "features" , "Label")

    //Build Regession Model
    val m = learnRegression(bdf)

    //Determine the next hyperparameter configuration
    getNextParameters(m, Allparam_df, df_data , step, classifiermgr,classifier)
  }

  /**
    * this function build a regression Model based on the paramters configuration
    * and their accuracy
    * the dataframe here is the hyperparameter + accuracy
    * @param df
    * @return
    */
  def learnRegression(df:DataFrame): RandomForestRegressionModel =
  {
    val rf = new RandomForestRegressor()
      .setLabelCol("Label")
      .setFeaturesCol("features")
    val model = rf.fit(df)
    return model
  }

  /**
    * this function do the following:
    * 1- use the regression model to predict the accuracy for each Hyperparameter
    * 2- select best hyperparameter (based on the expected Accuracy)
    * 3 - check if i tried this hyperparameter configuration before (if yes i should get another one)
    * 4- convert the selected hyperparameter to ParamMap
    * 5- train the classifier based on the selected hyperparam and add the result to a
    * global list "ParamMapAccuracyLst"
    * @param model
    * @param df_Param
    * @param df_data
    * @param step
    * @param classifiermgr
    * @param ClassifierName
    */
  def getNextParameters(model:RandomForestRegressionModel , df_Param:DataFrame , df_data:DataFrame , step:Int, classifiermgr:ClassifiersManager , ClassifierName:String )
  {
    var exist = false
    var accuracy = 0.0
    var acc = 0.0

    val predictions = model.transform(df_Param)
    var predicted_Param_accu = predictions.orderBy(desc("prediction")).collect().toList
    var currentVector: org.apache.spark.ml.linalg.DenseVector = null

    breakable {
      for (i <- 0 to predicted_Param_accu.size - 1) {
        exist = false
        var currentParam = predicted_Param_accu(i)(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector]
        ParamMapAccuracyLst.foreach(e => {
          exist=  compareRowtoParamMap(currentParam ,e._1 , ClassifierName )
        })
        if(! exist) {
          currentVector = currentParam
          acc = predicted_Param_accu(i)(1).asInstanceOf[Double]
          break
        }
      }
    }

    println( "Selected Paramters: " + currentVector.toArray.toList + " expected accuracy:" + acc)
    var pm = convertVectortoParamMap(currentVector ,ClassifierName)

    var res = singleLearn(df_data, pm, step, ClassifierName, classifiermgr)
    accuracy = res._2

    if(ClassifierName == "RandomForestClassifier")
      ParamMapAccuracyLst.append( (pm ,accuracy ,res._3.asInstanceOf[RandomForestClassificationModel]))
    if(ClassifierName == "LogisticRegression")
      ParamMapAccuracyLst.append( (pm ,accuracy ,res._3.asInstanceOf[LogisticRegressionModel]))
    if(ClassifierName == "DecisionTreeClassifier")
      ParamMapAccuracyLst.append( (pm ,accuracy ,res._3.asInstanceOf[DecisionTreeClassificationModel]))
    if( ClassifierName == "GBTClassifier")
      ParamMapAccuracyLst.append( (pm ,accuracy ,res._3.asInstanceOf[GBTClassificationModel]))
    if( ClassifierName == "LinearSVC")
      ParamMapAccuracyLst.append( (pm ,accuracy ,res._3.asInstanceOf[LinearSVCModel]))
    if(ClassifierName == "MultilayerPerceptronClassifier")
      ParamMapAccuracyLst.append( (pm ,accuracy ,res._3.asInstanceOf[MultilayerPerceptronClassificationModel]))

    ParamMapAccuracyLst.append( (pm ,accuracy ,null ) )

  }

  /**
    * this function convert Vector of parameters to ParamMap (based on the classifier)
    * @param inputVector
    * @param ClassifierName
    * @return
    */
  def convertVectortoParamMap(inputVector:org.apache.spark.ml.linalg.DenseVector , ClassifierName:String): ParamMap =
  {
    var pm = new ParamMap()
    var currentParam = inputVector.toArray
    if (ClassifierName == "RandomForestClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var numTrees = currentParam(2)
      var maxDepth = currentParam(3)
      var minInfoGain = currentParam(4)

      val rf = new RandomForestClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(rf.maxBins, maxBins.toInt)
      pm.put(rf.minInstancesPerNode, minInstancesPerNode.toInt)
      pm.put(rf.numTrees, numTrees.toInt)
      pm.put(rf.maxDepth, maxDepth.toInt)
      pm.put(rf.minInfoGain, minInfoGain.toDouble)
      pm.put(rf.impurity, "gini")
    }

    if (ClassifierName == "LogisticRegression"){
      var fitIntercept = false
      if(currentParam(0) == 1.0)
        fitIntercept = true
      var maxIter = currentParam(1)
      var regParam = currentParam(2)
      var elasticNetParam = currentParam(3)
      var standardization = false
      if(currentParam(4) == 1.0)
        standardization = true
      var tol = currentParam(5)

      val lr = new LogisticRegression()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(lr.fitIntercept , fitIntercept)
      pm.put(lr.maxIter , maxIter.toInt)
      pm.put(lr.regParam ,regParam )
      pm.put(lr.elasticNetParam ,elasticNetParam )
      pm.put(lr.standardization ,standardization )
      pm.put(lr.tol , tol)
    }

    if (ClassifierName == "DecisionTreeClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var maxDepth = currentParam(2)
      var minInfoGain = currentParam(3)

      val dt = new DecisionTreeClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(dt.maxBins, maxBins.toInt)
      pm.put(dt.minInstancesPerNode, minInstancesPerNode.toInt)
      pm.put(dt.maxDepth, maxDepth.toInt)
      pm.put(dt.minInfoGain, minInfoGain.toDouble)
      pm.put(dt.impurity, "gini")
    }

    if (ClassifierName == "GBTClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var maxDepth = currentParam(2)
      var minInfoGain = currentParam(3)

      val gbt = new GBTClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(gbt.maxBins, maxBins.toInt)
      pm.put(gbt.minInstancesPerNode, minInstancesPerNode.toInt)
      pm.put(gbt.maxDepth, maxDepth.toInt)
      pm.put(gbt.minInfoGain, minInfoGain.toDouble)
      pm.put(gbt.impurity, "gini")
    }

    if( ClassifierName == "LinearSVC"){
      var maxIter = currentParam(0)
      var regParam = currentParam(1)
      val svc = new LinearSVC()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(svc.maxIter , maxIter.toInt)
      pm.put(svc.regParam ,regParam )

    }

    if( ClassifierName == "MultilayerPerceptronClassifier"){
      var maxIter = currentParam(0)
      var Layers:Array[Int] =  Array(currentParam(1).toInt,currentParam(2).toInt,currentParam(3).toInt)
      val mlp = new MultilayerPerceptronClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(mlp.maxIter , maxIter.toInt)
      pm.put(mlp.layers ,Layers )

    }

    return pm
  }

  /**
    * compare Vector of parameter against PAramMap
    * i used this function to check if i used this hyperparameter before or not
    * @param inputVector
    * @param pm
    * @param ClassifierName
    * @return
    */
  def compareRowtoParamMap(inputVector:org.apache.spark.ml.linalg.DenseVector , pm:ParamMap, ClassifierName:String ): Boolean =
  {
    var currentParam = inputVector.toArray
    var result = false
    if (ClassifierName == "RandomForestClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var numTrees = currentParam(2)
      var maxDepth = currentParam(3)
      var minInfoGain = currentParam(4)

      var maxBins_b = 0
      var minInstancesPerNode_b = 0
      var numTrees_b = 0
      var maxDepth_b = 0
      var minInfoGain_b = 0.0

      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "numTrees")
          numTrees_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain_b = parmmap.value.asInstanceOf[Double]
      }

      if( maxBins == maxBins_b  &&
        minInstancesPerNode == minInstancesPerNode_b &&
        numTrees == numTrees_b &&
        maxDepth == maxDepth_b &&
        minInfoGain == minInfoGain_b)
        result = true
    }
    if (ClassifierName == "LogisticRegression") {
      var fitIntercept = false
      if(currentParam(0) == 1.0)
        fitIntercept = true
      var maxIter = currentParam(1)
      var regParam = currentParam(2)
      var elasticNetParam = currentParam(3)
      var standardization = false
      if(currentParam(4) == 1.0)
        standardization = true
      var tol = currentParam(5)

      var fitIntercept_b = false
      var maxIter_b = 0
      var regParam_b = 0.0
      var elasticNetParam_b = 0.0
      var standardization_b = false
      var tol_b = 0.0

      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "fitIntercept")
          fitIntercept_b = parmmap.value.asInstanceOf[Boolean]
        if (parmmap.param.name == "maxIter")
          maxIter_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "regParam")
          regParam_b = parmmap.value.asInstanceOf[Double]
        if (parmmap.param.name == "elasticNetParam")
          elasticNetParam_b = parmmap.value.asInstanceOf[Double]
        if (parmmap.param.name == "standardization")
          standardization_b = parmmap.value.asInstanceOf[Boolean]
        if (parmmap.param.name == "tol")
          tol_b = parmmap.value.asInstanceOf[Double]
      }

      if( fitIntercept == fitIntercept_b  &&
        maxIter == maxIter_b &&
        regParam == regParam_b &&
        elasticNetParam == elasticNetParam_b &&
        standardization == standardization_b &&
        tol == tol_b
      )
        result = true

    }
    if (ClassifierName == "DecisionTreeClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var maxDepth = currentParam(2)
      var minInfoGain = currentParam(3)

      var maxBins_b = 0
      var minInstancesPerNode_b = 0
      var maxDepth_b = 0
      var minInfoGain_b = 0.0

      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain_b = parmmap.value.asInstanceOf[Double]
      }

      if( maxBins == maxBins_b  &&
        minInstancesPerNode == minInstancesPerNode_b &&
        maxDepth == maxDepth_b &&
        minInfoGain == minInfoGain_b)
        result = true
    }
    if (ClassifierName == "GBTClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var maxDepth = currentParam(2)
      var minInfoGain = currentParam(3)

      var maxBins_b = 0
      var minInstancesPerNode_b = 0
      var maxDepth_b = 0
      var minInfoGain_b = 0.0

      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain_b = parmmap.value.asInstanceOf[Double]
      }

      if( maxBins == maxBins_b  &&
        minInstancesPerNode == minInstancesPerNode_b &&
        maxDepth == maxDepth_b &&
        minInfoGain == minInfoGain_b)
        result = true
    }
    if (ClassifierName == "LinearSVC") {
      var maxIter = currentParam(0)
      var regParam = currentParam(1)

      var maxIter_b = 0
      var regParam_b = 0.0


      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)

        if (parmmap.param.name == "maxIter")
          maxIter_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "regParam")
          regParam_b = parmmap.value.asInstanceOf[Double]

      }

      if(
        maxIter == maxIter_b &&
          regParam == regParam_b
      )
        result = true

    }
    if (ClassifierName == "MultilayerPerceptronClassifier") {
      var maxIter = currentParam(0).toInt
      var Layer = currentParam(2).toInt
      var maxIter_b = 0
      var Layer_b = 0
      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "maxIter")
          maxIter_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "layers") {
          Layer_b = parmmap.value.asInstanceOf[Array[Int]](1)
        }
      }

      if(
        maxIter == maxIter_b &&
          Layer == Layer_b
      )
        result = true

    }

    return result

  }

  /**
    * Train classifier on a percentage of data based on the sent ParamMap
    * @param df_data
    * @param paramMap
    * @param step
    * @param ClassifierName
    * @param ClassifierMgr
    * @return
    */
  def singleLearn(df_data: Dataset[_], paramMap:ParamMap , step:Int, ClassifierName:String , ClassifierMgr:ClassifiersManager ): (ParamMap, Double, Model[_]) =
  {
    //println("Start singleLearn")
try {
    var accuracy = 0.0
    var percent = $(bayesianStepDataPercentage) * (step+1)
    if(percent > 0.8)
      percent = 0.8
    val Array(trainingDataset, validationDataset) = df_data.randomSplit(Array(percent, 1- percent), 1234)
    var rf = createClassifier(paramMap , ClassifierName)

    if (!IsTimeOut()) {
      var rf_model = rf.fit(trainingDataset).asInstanceOf[Model[_]]
      if (!IsTimeOut()) {
        val rf_predictions = rf_model.transform(validationDataset)
        val evaluator = ClassifierMgr.evaluator
        accuracy = evaluator.evaluate(rf_predictions)
        println("      - Bayesian Step " + step + " (on " + formatter.format(percent * 100) + "% of the Data),Acc:" + (formatter.format(accuracy * 100)).toString + "%")

        if (ClassifierName == "RandomForestClassifier")
          return (paramMap, accuracy, rf_model.asInstanceOf[RandomForestClassificationModel])
        if (ClassifierName == "LogisticRegression")
          return (paramMap, accuracy, rf_model.asInstanceOf[LogisticRegressionModel])
        if (ClassifierName == "DecisionTreeClassifier")
          return (paramMap, accuracy, rf_model.asInstanceOf[DecisionTreeClassificationModel])
        if (ClassifierName == "GBTClassifier")
          return (paramMap, accuracy, rf_model.asInstanceOf[GBTClassificationModel])
        if (ClassifierName == "LinearSVC")
          return (paramMap, accuracy, rf_model.asInstanceOf[LinearSVCModel])
        if (ClassifierName == "MultilayerPerceptronClassifier")
          return (paramMap, accuracy, rf_model.asInstanceOf[MultilayerPerceptronClassificationModel])
       }
      else
        println("        TimeOut")
      }
    else
      println("        TimeOut")
    return (paramMap , accuracy , null)
} catch {
  case ex: Exception =>
    println("Exception (Hyperband - " + ClassifierName + "- learn): " + ex.getMessage)
    ex.printStackTrace()
    return null
}


  }

  /**
    * Create a classifier object using the sent parameter Map
    * @param paramMap
    * @param ClassifierName
    * @return
    */
  def createClassifier(paramMap:ParamMap , ClassifierName:String): Estimator[_] =
  {
    //println("Start createClassifier")
    var estimator:Estimator[_] = null

    if( ClassifierName == "RandomForestClassifier") {
      var maxBins = 0
      var minInstancesPerNode = 0
      var numTrees = 0
      var maxDepth = 0
      var minInfoGain = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "numTrees")
          numTrees = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain = parmmap.value.asInstanceOf[Double]
      }

      val rf = new RandomForestClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMinInstancesPerNode(minInstancesPerNode)
        .setMinInfoGain(minInfoGain)
        .setMaxDepth(maxDepth)
        .setNumTrees(numTrees)
        .setMaxBins(maxBins)
      estimator = rf
      //println("minInstancesPerNode:" + minInstancesPerNode + ", minInfoGain:" + minInfoGain + " ,:setMaxDepth" + maxDepth + " ,:numTrees" + numTrees + " ,:maxBins" + maxBins)
    }

    if( ClassifierName == "LogisticRegression") {
      var fitIntercept = false
      var maxIter = 0
      var regParam = 0.0
      var elasticNetParam = 0.0
      var standardization = false
      var tol = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "fitIntercept")
          fitIntercept = parmmap.value.asInstanceOf[Boolean]
        if (parmmap.param.name == "maxIter")
          maxIter = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "regParam")
          regParam = parmmap.value.asInstanceOf[Double]
        if (parmmap.param.name == "elasticNetParam")
          elasticNetParam = parmmap.value.asInstanceOf[Double]
        if (parmmap.param.name == "standardization")
          standardization = parmmap.value.asInstanceOf[Boolean]
        if (parmmap.param.name == "tol")
          tol = parmmap.value.asInstanceOf[Double]
      }
      val lr = new LogisticRegression()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setFitIntercept(fitIntercept)
        .setMaxIter(maxIter)
        .setRegParam(regParam)
        .setElasticNetParam(elasticNetParam)
        .setStandardization(standardization)
        .setTol(tol)
      estimator = lr
    }

    if( ClassifierName == "DecisionTreeClassifier") {
      var maxBins = 0
      var minInstancesPerNode = 0
      var maxDepth = 0
      var minInfoGain = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain = parmmap.value.asInstanceOf[Double]
      }

      val dt = new DecisionTreeClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMinInstancesPerNode(minInstancesPerNode)
        .setMinInfoGain(minInfoGain)
        .setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
      estimator = dt

    }

    if ( ClassifierName == "GBTClassifier") {

      var maxBins = 0
      var minInstancesPerNode = 0
      var maxDepth = 0
      var minInfoGain = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain = parmmap.value.asInstanceOf[Double]
      }

      val gbt = new GBTClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMinInstancesPerNode(minInstancesPerNode)
        .setMinInfoGain(minInfoGain)
        .setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
      estimator = gbt

    }

    if( ClassifierName == "LinearSVC") {
      var maxIter = 0
      var regParam = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxIter")
          maxIter = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "regParam")
          regParam = parmmap.value.asInstanceOf[Double]
      }
      val svc= new LinearSVC()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMaxIter(maxIter)
        .setRegParam(regParam)
      estimator = svc

    }

    if ( ClassifierName == "MultilayerPerceptronClassifier")  {
      var maxIter = 0
      var Layers :Array[Int]= null

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxIter")
          maxIter = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "layers")
          Layers = parmmap.value.asInstanceOf[Array[Int]]
      }
      val mlp= new MultilayerPerceptronClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMaxIter(maxIter)
        .setLayers(Layers)
      estimator = mlp
    }

    return estimator
  }

}

/**
  *
  */
object BayesianOpt extends MLReadable[BayesianOpt] {

  @Since("2.0.0")
  override def read: MLReader[BayesianOpt] = new BayesianOptReader

  @Since("2.0.0")
  override def load(path: String): BayesianOpt = super.load(path)

  private[BayesianOpt] class BayesianOptWriter(instance: BayesianOpt)
    extends MLWriter {

    ValidatorParams.validateParams(instance)

    override protected def saveImpl(path: String): Unit =
      ValidatorParams.saveImpl(path, instance, sc)
  }

  private class BayesianOptReader extends MLReader[BayesianOpt] {

    /** Checked against metadata when loading model */
    private val className = classOf[BayesianOpt].getName

    override def load(path: String): BayesianOpt = {
      implicit val format = DefaultFormats

      val (metadata, estimator, evaluator, estimatorParamMaps) =
        ValidatorParams.loadImpl(path, sc, className)
      val tvs = new BayesianOpt(metadata.uid)
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
  * Model from train BayesianOpt.
  *
  * @param uid Id.
  * @param bestModel Estimator determined best model.
  * @param validationMetrics Evaluated validation metrics.
  */
@Since("1.5.0")
class BayesianOptModel  (
                        @Since("1.5.0") override val uid: String,
                        @Since("1.5.0") val bestModel: Model[_],
                        @Since("1.5.0") val validationMetrics: Array[Double]) extends Model[BayesianOptModel] with BayesianOptParams with MLWritable {

  /** A Python-friendly auxiliary constructor. */
  private[ml] def this(uid: String, bestModel: Model[_], validationMetrics: JList[Double]) = {
    this(uid, bestModel, validationMetrics.asScala.toArray)
  }

  private var _subModels: Option[Array[Model[_]]] = None

  private[tuning] def setSubModels(subModels: Option[Array[Model[_]]])
  : BayesianOptModel = {
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
  override def copy(extra: ParamMap): BayesianOptModel = {
    val copied = new BayesianOptModel (
      uid,
      bestModel.copy(extra).asInstanceOf[Model[_]],
      validationMetrics.clone()
    ).setSubModels(BayesianOptModel.copySubModels(_subModels))
    copyValues(copied, extra).setParent(parent)
  }

  @Since("2.0.0")
  override def write: BayesianOptModel.BayesianOptModelWriter = {
    new BayesianOptModel.BayesianOptModelWriter(this)
  }
}



object BayesianOptModel extends MLReadable[BayesianOptModel] {

  private[BayesianOptModel] def copySubModels(subModels: Option[Array[Model[_]]])
  : Option[Array[Model[_]]] = {
    subModels.map(_.map(_.copy(ParamMap.empty).asInstanceOf[Model[_]]))
  }

  @Since("2.0.0")
  override def read: MLReader[BayesianOptModel] = new BayesianOptModelReader

  @Since("2.0.0")
  override def load(path: String): BayesianOptModel = super.load(path)

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
  final class BayesianOptModelWriter private[tuning] (
                                                     instance: BayesianOptModel) extends MLWriter {

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

  private class BayesianOptModelReader extends MLReader[BayesianOptModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[BayesianOptModel].getName

    override def load(path: String): BayesianOptModel = {
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

      val model = new BayesianOptModel(metadata.uid, bestModel, validationMetrics)
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