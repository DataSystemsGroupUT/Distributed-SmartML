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

object App_GetBestModel {


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


    // Dataset
    val i = args(0).toInt
    // type (Grid Search, TransmogrifAI, Random Search, DSmart ML)
    val j = args(1).toInt
    // Time Limit
    val t = args(2).toInt
    // Parallelism
    val p = 3 //args(3).toInt

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



    // Set Log Level to error only (don't show warning)
    spark.sparkContext.setLogLevel("ERROR")






    //for ( i <- Seq(2,8,15,16,17,19,23,42,46,48,52,54) ) {
    //  for ( i <- Seq(48,59,69,74,76) ) {
    //Create Logger Instance
    var logger = new Logger(logpath)
    try{

      // Load Dataset
      var dataloader = new DataLoader(spark, i, dataFolderPath, logger)
      var rawdata = dataloader.getData()

      logger.logOutput("DSmart ML 1.0 => Data set number :" + i + "\n")
      logger.logOutput("============================================================ \n")
      //var rawdata = ExampleDataset.getDataset(spark, dataFolderPath , i)

      // get best Model for this dataset using Distributed SmartML Library
      //===================================================================================
      if (j == 1) {
         //for (ti <- Array(20,40,60,80,100) ){
        try {
          // Set Start Time
          val starttime1 = new java.util.Date().getTime
          println("--------------Smart ML-------------------------------------")
          //logger.logOutput("--------------Smart ML------------------------------------- \n")
          var mselector = new ModelSelector(spark, logger, eta = 3, maxResourcePercentage = 100, HP_MaxTime = t, Parallelism = p, TryNClassifier = 6)
          var res = mselector.getBestModel(rawdata)
          println(res._2._2.toString())
          val Endtime1 = new java.util.Date().getTime
          val TotalTime1 = Endtime1 - starttime1
          //logger.logOutput("--------------------------------------------------------------------- \n")
          //logger.logOutput("Result \n")
          //logger.logOutput("--Best Algorithm:" + res._1 + " \n")
          //logger.logOutput("--Accuracy:" + res._2._3 + " \n")
          //logger.logOutput("--Total Time:" + (TotalTime1 / 1000.0).toString + " \n")
          println("  =>Result:")
          println("  |--Best Algorithm:" + res._1)
          println("  |--Accuracy:" + res._2._3)
          println("  |--Total Time:" + (TotalTime1 / 1000.0).toString)
          //if (res._2._2 != null)
          //  logger.logOutput("Parameters:" + res._2._2 + " \n")
          //logger.logOutput(i + "," + res._1 + "," + t + "," + (TotalTime1 / 1000.0).toString + "," + res._2._3 + "\n")
          //}
        }
        catch {
          case ex: Exception => println(ex.getMessage)
          //                         logger.logOutput("Exception: " + ex.getMessage)
          // logger.close()
        }
      //}
      }
        // TransmogrifAI
        //===================================================================================
      if( j == 2) {
        println("------------------TransmogrifAI -------------------------------------")
        val starttime3 = new java.util.Date().getTime
        // Extract response and predictor Features
        rawdata = rawdata.withColumn(TargetCol + "_", col(TargetCol).cast(DoubleType))
        rawdata = rawdata.drop(TargetCol)
        rawdata = rawdata.withColumnRenamed(TargetCol + "_", TargetCol)
        val Array(trainingData, testData) = rawdata.randomSplit(Array(0.8, 0.2), 1234)
        val (label, predictors) = FeatureBuilder.fromDataFrame[RealNN](trainingData, response = TargetCol)

        // Automated feature engineering
        val featureVector = predictors.transmogrify()

        //val featureVector = DataLoader.convertDFtoVecAssembly(rawdata, "features" ,TargetCol )
         val modelSelector = MultiClassificationModelSelector.withTrainValidationSplit(
           splitter = Option(DataCutter(reserveTestFraction = 0.2, seed = 10L)),
           validationMetric = Evaluators.MultiClassification.error(),
                     seed = 10L)
           .setInput(label, featureVector)
         val pred =  modelSelector.getOutput()

        // Setting up a TransmogrifAI workflow and training the model
         val model = new OpWorkflow().setInputDataset(trainingData).setResultFeatures(pred).train()
         model.setInputRDD(testData.rdd)

         val evaluator = Evaluators.MultiClassification()
                  .setLabelCol(TargetCol)
                  .setPredictionCol(pred)

         val (transformedTrainData, metrics) = model.scoreAndEvaluate(evaluator = evaluator)

         val Endtime3 = new java.util.Date().getTime
         val TotalTime3 = Endtime3 - starttime3
         val ind = model.summaryPretty().indexOf("Selected Model - ")
         logger.logOutput("------------------TransmogrifAI ------------------------------------- \n")
         logger.logOutput("Total Time:" + TotalTime3.toString + " \n")
         logger.logOutput("Best Algorithm:" + model.summaryPretty().substring(ind + 17, ind + 17 + 25) + " \n")
         logger.logOutput("Accuracy:" + (1 - metrics.toMap("Error").asInstanceOf[Double]) + " \n")

         println("  =>Result:")
         println("  |-- Total Time:" + (TotalTime3 / 1000.0).toString )
         println("  |-- Best Algorithm:" + model.summaryPretty().substring(ind + 17, ind + 17 + 25) )
         println("  |-- Accuracy:" + (1 - metrics.toMap("Error").asInstanceOf[Double]) )

        println(model.summaryPretty())
        logger.logOutput("\n")
        logger.logOutput("\n")

        }



      // Grid Search
      //===================================================================================
      if (j == 3) {
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
        //                         logger.logOutput("Exception: " + ex.getMessage)
        logger.close()
    }
    //println("Model summary:\n" + model.summaryPretty())
    logger.close()
    //}
  }


}
