package org.dsmartml.examples

import com.salesforce.op.OpWorkflow
import com.salesforce.op.evaluators.Evaluators
import com.salesforce.op.features.FeatureBuilder
import com.salesforce.op.features.types.RealNN
import com.salesforce.op.stages.impl.classification.MultiClassificationModelSelector
import com.salesforce.op.stages.impl.tuning.DataCutter
import org.apache.spark.ml.classification.RandomForestClassifier
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.mllib.tree.RandomForest
import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.dsmartml._

object App_Test {


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



    val i = args(0).toInt
    //val j = args(1).toInt
    //val t = 100//args(2).toInt
    //val p = 3 //args(3).toInt

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



    var featuresCol = "features"


    //for ( i <- Seq(4,5,7,10,13,25,30,48,59,69,74,76) ) {
    //  for ( i <- Seq(48,59,69,74,76) ) {
    //Create Logger Instance
    var logger = new Logger(logpath)
    try{

      // Load Dataset
      var dataloader = new DataLoader(spark, i, dataFolderPath, logger)
      var rawdata = dataloader.getData()
      logger.logOutput("Data set number :" + i + "\n")
      logger.logOutput("============================================================ \n")
      //var rawdata = ExampleDataset.getDataset(spark, dataFolderPath , i)
      var dataset = DataLoader.convertDFtoVecAssembly(rawdata, "features",TargetCol )
      val Array(train, test) = dataset.randomSplit(Array(0.8, 0.2))

      val rf = new RandomForestClassifier()
        .setLabelCol(TargetCol)
        .setFeaturesCol(featuresCol)
        .setImpurity("gini")
        .setMaxBins(32)
        .setNumTrees(62)
        .setMaxDepth(13)
        .setMinInfoGain(0.001)
        .setMinInstancesPerNode(1)
      val model = rf.fit(train)
      val pre = model.transform(test)

      val evaluator = new MulticlassClassificationEvaluator()
        .setLabelCol(TargetCol)
        .setPredictionCol("prediction")
        .setMetricName("accuracy")

      val metric = evaluator.evaluate(pre)
      println("     - Accuracy:" + metric )

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
