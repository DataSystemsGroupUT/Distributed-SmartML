package org.dsmartml.examples

import com.salesforce.op.OpWorkflow
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types.RealNN
import com.salesforce.op.stages.impl.classification.MultiClassificationModelSelector
import com.salesforce.op.stages.impl.tuning.DataCutter
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.dsmartml._

object App_KDE {


  def main(args: Array[String]): Unit = {



    // Set File pathsd
    //*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*
    //1- Google Cloud -->
    //-------------------------------------------------
    //var dataFolderPath = "gs://sparkstorage/datasets/"
    //var logpath = "/home/eissa_abdelrahman5/"

    //2- Local -->
    //-------------------------------------------------
    var dataFolderPath = "/media/eissa/New/data/"
    var logpath = "/media/eissa/New/data/"

    //3- Azure Cloud -->
    //-------------------------------------------------
    //var dataFolderPath = "wasb:///example/data/"
    //var logpath = "/home/sshuser/"





    var TargetCol = "y"

    // Create Spark Session
    implicit val spark = SparkSession
      .builder()
      .appName("Distributed Smart ML 1.0")
      //.config("packages" , "com.salesforce.transmogrifai:transmogrifai-core_2.11:0.5.3")
      .config("spark.master", "local")
      //.config("spark.master", "yarn")
      //.config("spark.executor.memory", "15g")
      //.config("spark.driver.memory", "6g")
      //.config("spark.storage.memoryFraction" , "2")
      //.config("spark.driver.cores" , 2)
      //.config("spark.executor.cores" , 7)
      .getOrCreate()



    // Set Log Level to error only (don't show warning)
    spark.sparkContext.setLogLevel("ERROR")



    val data = List("john","paul","george","ringo")

    val dataRDD = spark.sparkContext.makeRDD(data)

    val scriptPath =  dataFolderPath + "test.py"

    val pipeRDD = dataRDD.pipe(scriptPath)

    pipeRDD.foreach(println)


    //for ( i <- Seq(4,5,7,10,13,25,30,48,59,69,74,76) ) {
    //  for ( i <- Seq(48,59,69,74,76) ) {
    //Create Logger Instance
    var logger = new Logger(logpath)
    try{

      // Load Dataset
      //var dataloader = new DataLoader(spark, i, dataFolderPath, logger)
      //var rawdata = dataloader.getData()

    }
    catch{
      case ex:Exception => println(ex.getMessage)
        //                         logger.logOutput("Exception: " + ex.getMessage)
        logger.close()
    }
    //println("Model summary:\n" + model.summaryPretty())
    logger.close()
    //}
  }


}
