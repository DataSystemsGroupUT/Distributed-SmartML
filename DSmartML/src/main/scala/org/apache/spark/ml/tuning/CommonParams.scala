package org.apache.spark.ml.tuning

import org.apache.spark.ml.Model
import org.apache.spark.ml.param._

trait CommonParams extends ValidatorParams{

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
  * Parameters for the optimization process result
  */
trait OptimizerResult
{
  var bestParam :ParamMap = null
  var bestModel :Model[_] = null
  var bestmetric :Double = 0.0
}
