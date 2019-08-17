package org.dsmartml.examples

import breeze.stats.distributions._
import com.salesforce.op.OpWorkflow
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types.RealNN
import com.salesforce.op.stages.impl.classification.MultiClassificationModelSelector
import com.salesforce.op.stages.impl.tuning.DataCutter
import org.apache.commons.math3.random.MersenneTwister
import org.apache.spark.ml.classification.LogisticRegression
import org.apache.spark.ml.tuning.{RandomSearch, RandomSearchManager}
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.dsmartml._

object App_RandomSearch {


  def main(args: Array[String]): Unit = {


    // Set File pathsd
    //*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //1- Google Cloud -->
    //-------------------------------------------------
    var dataFolderPath = "gs://sparkstorage/datasets/"
    var logpath = "/home/eissa_abdelrahman5/"

    //2- Local -->
    //-------------------------------------------------
    //var dataFolderPath = "/media/eissa/New/data/"
    //var logpath = "/media/eissa/New/data/"

    //3- Azure Cloud -->
    //-------------------------------------------------
    //var dataFolderPath = "wasb:///example/data/"
    //var logpath = "/home/sshuser/"

    //implicit val randBasis: RandBasis = new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(1234)))

    def setSeed(seed: Int = 0) = {
      new RandBasis(new ThreadLocalRandomGenerator(new MersenneTwister(seed)))
    }
    //println(Gamma (0.5, 0.1).sample(100))
    //println(Poisson(50)(setSeed(123)).sample(100))
    //println("===================================================")
    //println(Poisson(50)(setSeed(123)).sample(100))
    //println(Binomial(5 , 6.0).sample(100))

    // dataset
    val i = args(0).toInt
    // time limit
    val t = args(1).toInt
    // parallelism
    val p = args(2).toInt

    var TargetCol = "y"

    // Create Spark Session
    implicit val spark = SparkSession
      .builder()
      .appName("Distributed Smart ML 1.0")
      //.config("packages" , "com.salesforce.transmogrifai:transmogrifai-core_2.11:0.5.3")
      //.config("spark.master", "local")
      .config("spark.master", "yarn")
      //.config("spark.executor.memory", "15g")
      //.config("spark.driver.memory", "6g")
      //.config("spark.storage.memoryFraction" , "2")
      //.config("spark.driver.cores" , 2)
      //.config("spark.executor.cores" , 7)
      .getOrCreate()



    //println(Poisson(50)(randBasis))

    // Set Log Level to error only (don't show warning)
    spark.sparkContext.setLogLevel("ERROR")

    var logger = new Logger(logpath)
    //for ( i <- Seq(2,8,15,16,17,19,23,42,46,48,52,54) ) {

    var dataloader = new DataLoader(spark, i, dataFolderPath, logger)
    var rawdata = dataloader.getData()
    var dataset = DataLoader.convertDFtoVecAssembly(rawdata, "features", TargetCol)
    dataset.persist()
    for (time <- Array(50, 100, 200, 300, 400)) {
      try {
        val starttime1 = new java.util.Date().getTime
        var rs = new RandomSearchManager(ParamNumber = 125, MaxTime = time, MaxResource = 100, sp = spark, Parallelism = p)
        val model = rs.fit(dataset)
        val Endtime1 = new java.util.Date().getTime
        val TotalTime1 = Endtime1 - starttime1
        println(rs.bestParam)
        println("  =>Result:")
        println("  |--Best Algorithm:" + rs.bestClassifier)
        println("  |--Accuracy:" + rs.bestmetric)
        println("  |--Total Time:" + (TotalTime1 / 1000.0).toString)
        println("=======================================================")

        var plist = ""
        for (d <- rs.bestParam.toSeq.toList) {
          if (plist != "")
            plist = d.param.name + ":" + d.value + "," + plist
          else
            plist = d.param.name + ":" + d.value
        }
        logger.logOutput("Random Search => Data set number :" + i + "\n")
        logger.logOutput("============================================================ \n")
        logger.logOutput("Max Time:" + time + ", Total Time:" + (TotalTime1 / 1000.0).toString + ", Accuracy:" + rs.bestmetric + ", Parameters:(" + plist + ") \n")

      } catch {
        case ex: Exception => println(ex.getMessage)
          //                         logger.logOutput("Exception: " + ex.getMessage)
          logger.close()
      }
    }
      logger.close()
 // }
    //trainingData_MinMaxScaled)
    //var accuracy = rs.bestmetric

    //println("best Accuracy:" +accuracy )


  }


}
