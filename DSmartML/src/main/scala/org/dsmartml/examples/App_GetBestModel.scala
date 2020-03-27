package org.dsmartml.examples

import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.dsmartml._

import scala.collection.mutable.ListBuffer

object App_GetBestModel {


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


    // Dataset
    val i = args(0).toInt
    // type ( 1:Random Search, 2:Hyperband, 3:Bayesian Opt, 5:Grid Search, )
    val j = args(1).toInt
    // Time Limit
    val t = args(2).toInt
    // skip sh
    val skip_SH = args(3).toInt
    //SplitbyClass
    val SplitbyClass =  if (args(4).toInt == 1 ) true else false
    //basicDataPerventage
    val basicDataPercentage = args(5).toDouble

    // Parallelism
    val p = 3 //args(3).toInt

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

    //Create Logger Instance
    var logger = new Logger(logpath)
    var Classifiers = "RandomForestClassifier,LogisticRegression,DecisionTreeClassifier,MultilayerPerceptronClassifier,LinearSVC,NaiveBayes,GBTClassifier,LDA,QDA"

    // Load Dataset
    var dataloader = new DataLoader(spark, i, dataFolderPath, logger)
    var rawdata = dataloader.getData()

    println("Number of partations(after loading): " + rawdata.rdd.getNumPartitions)
    try{

      // get best Model for this dataset using Distributed SmartML Library
      //===================================================================================
      if (j == 1 || j == 2 || j == 3) {
        //for (ti <- Array(20,40,60,80,100) ){
        try {
          var mselector = new ModelSelector(  spark,
                                              logpath,
                                              eta = 3,
                                              maxResourcePercentage = 100,
                                              HP_MaxTime = t,
                                              HPOptimizer = j,
                                              skip_SH = skip_SH,
                                              SplitbyClass = SplitbyClass,
                                              basicDataPercentage = basicDataPercentage ,
                                              Classifiers = Classifiers
                                            )
          var res = mselector.getBestModel(rawdata)
        }
        catch {
          case ex: Exception => println(ex.getMessage)
        }
      }

      // Grid Search
      //===================================================================================
      if (j == 5) {
        println("Grid Search")
        val starttime2 = new java.util.Date().getTime
        logger.logOutput("============= Grid Search============================================= \n")
        var grdSearchMgr: GridSearchManager = new GridSearchManager()
        var res2 = grdSearchMgr.Search(spark, rawdata)
        logger.logOutput("--Best Algorithm:" + res2._1 + " \n")
        logger.logOutput("--Accuracy:" + res2._2._3 + " \n")
        val Endtime2 = new java.util.Date().getTime
        val TotalTime2 = Endtime2 - starttime2
        logger.logOutput("--Total Time:" + (TotalTime2 / 1000.0).toString + " \n")
        if (res2._2._2 != null)
          logger.logOutput("Parameters:" + res2._2._2 + " \n")
        logger.logOutput("===================================================================== \n")
      }

    }
    catch{
      case ex:Exception => println(ex.getMessage)
      logger.close()
    }
    logger.close()
  }


}
