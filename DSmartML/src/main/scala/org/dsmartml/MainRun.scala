package org.dsmartml
import breeze.stats.distributions.{Poisson, RandBasis, ThreadLocalRandomGenerator}
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.sql.SparkSession
import org.dsmartml._

object Run_Main {

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
                            .config("spark.rpc.askTimeout", "600s")
		            //.config("spark.driver.memory","4g")
                            //.config("spark.executor.memory", "1g")
                            .getOrCreate()

    // Set Log Level to error only (don't show warning)
    spark.sparkContext.setLogLevel("ERROR")

    // path
    //=================================================================
    var dataPath = "/home/ubuntu/scala/data/"
    var logingpath = "/home/ubuntu/scala/logs/"
    val fileName = args(0)
    val time = args(1).toInt
    //val fileName = "blood.csv"
    // type (Grid Search, TransmogrifAI, Random Search, DSmart ML)

    // Read Data (from CSV file)
    //=================================================================
    var label = "class"
    var rawdata = spark.read.option("header",true)
                            .option("inferSchema","true")
                            .option("delimiter", ",")
                            .format("csv")
                            .load(fileName)
    rawdata = rawdata.withColumnRenamed("_c10" , label)

    // Find Best Model (Using DSmart ML Library)
    //=================================================================
    var mselector = new ModelSelector(spark, TargetCol = label,
                                      logpath = logingpath, HP_MaxTime = time,
                                      HPOptimizer = 2)
    var res = mselector.getBestModel(rawdata)
    spark.stop()
    println(res)
    //get the model
    //=================================================================
    //var model = res._2._1
    }






}
