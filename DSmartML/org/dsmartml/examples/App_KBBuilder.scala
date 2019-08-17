package org.dsmartml.examples

import org.apache.spark.sql.SparkSession
import org.dsmartml._

object App_KBBuilder {

  def main(args: Array[String]): Unit = {

    //Save Start Time
    val starttime =  new java.util.Date().getTime

    //var dataFolderPath = "gs://sparkstorage/datasets/"
    //var logpath = "/home/eissa_abdelrahman5/"

    var dataFolderPath = "/media/eissa/New/data/"
    var logpath = "/media/eissa/New/data/"

    val spark = SparkSession
      .builder()
      .appName("Java Spark SQL basic example")
      .config("spark.master", "local")
      //.config("spark.master", "yarn")
      .config("spark.executor.memory", "8g")
      .config("spark.driver.memory", "12g")
      //.config("spark.storage.memoryFraction" , "2")
      .config("spark.driver.cores" , 2)
      .config("spark.executor.cores" , 2)
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")
    var logger =  new Logger(logpath)

    val i = args(0).toInt
    val k = args(1).toInt

    for (j <- i to k) {

      var dataloader = new DataLoader(spark, j, dataFolderPath, logger)
      val df = dataloader.getData()
      var metadataMgr = new MetadataManager(spark , logger , "y")
      var mdata = metadataMgr.ExtractMetadata(df)
      var kbmgr = new KBManager(spark, logger ,"y" )
      kbmgr.SaveMetadata_toKB("dataFolderPath" , mdata)

      val Endtime = new java.util.Date().getTime
      val TotalTime = Endtime - starttime
      try {
        logger.logTime("Total Time:" + TotalTime.toString + "\n")
        println("Dataset:" + j + " Completed")
        println("=======================================================================")
      }catch{
        case ex:Exception => println(ex.getMessage)
          logger.close()
      }
    }

    logger.close()

  }


}
