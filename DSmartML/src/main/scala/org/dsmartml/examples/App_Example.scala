package org.dsmartml.examples
import breeze.stats.distributions.{Poisson, RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.sql.SparkSession
import org.dsmartml._

object App_Example {

  def setSeed(seed: Int = 0) = {
    new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
  }


  def main(args: Array[String]): Unit = {

    // Create Spark Session
    //=================================================================
    implicit val spark = SparkSession
                            .builder()
                            .appName("Distributed Smart ML 1.0")
                            .config("spark.master", "local")
                            .getOrCreate()
    // Set Log Level to error only (don't show warning)
    spark.sparkContext.setLogLevel("ERROR")

    // path
    //=================================================================
    var dataPath = "/media/eissa/New/data/"
    var logingpath = "/media/eissa/New/data/"

    // Read Data (from CSV file)
    //=================================================================
    var label = "y"
    var rawdata = spark.read.option("header",false)
                            .option("inferSchema","true")
                            .option("delimiter", ",")
                            .format("csv")
                            .load(dataPath + "07-avila.txt")
    rawdata = rawdata.withColumnRenamed("_c10" , label)

    // Find Best Model (Using DSmart ML Library)
    //=================================================================
    var mselector = new ModelSelector(spark, TargetCol = label,
                                      logpath = logingpath, HP_MaxTime = 100,
                                      HPOptimizer = 1)
    var res = mselector.getBestModel(rawdata)

    //get the model
    //=================================================================
    var model = res._2._1
    }






}
