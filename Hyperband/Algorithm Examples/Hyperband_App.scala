
import java.io.{File, PrintWriter}

import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}
import org.apache.spark.ml.tuning.{Hyperband, ParamGridBuilder, TrainValidationSplit}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.classification.{RandomForestClassificationModel, RandomForestClassifier}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator

object Hyperband_App {


  def main(args: Array[String]): Unit = {

    var dataFolderPath = "/home/eissa/mycode/data/"

    //Create Spark Session
    //==================================================================================================================
    val spark = SparkSession
      .builder()
      .appName("Java Spark SQL basic example")
      .config("spark.master", "local")
      .getOrCreate();

    // Prepare training and test data.
    //==================================================================================================================
    val rawdata = spark.read.option("header","true")
                  .option("inferSchema","true")
                 .format("csv")
                 .load(dataFolderPath + "Sensorless_drive_diagnosis.csv")
    var label = "y"

    val featurecolumns = rawdata.columns.filter(c => c != label)

    val assembler = new VectorAssembler()
      .setInputCols(featurecolumns)
      .setOutputCol("features_in")

    val scaler = new StandardScaler()
      .setInputCol("features_in")
      .setOutputCol("features")
      .setWithStd(true)
      .setWithMean(true)

    val Alldata = assembler.transform(rawdata.na.drop).select(label, "features_in")
    val scalermodel = scaler.fit(Alldata)
    var mydataset = scalermodel.transform(Alldata)
    val Array(trainingData, testData) = mydataset.randomSplit(Array(0.7, 0.3))


    // Create Estimator, Evaluator and Hyper parameters Grid
    //==================================================================================================================

    // 1- Estimator
    val rf = new RandomForestClassifier()
      .setLabelCol(label)
      .setFeaturesCol("features")

    val lr = new LinearRegression()


    // 2- Grid of Hyper- Parameters
    var paramGrid = new ParamGridBuilder()
      .addGrid(rf.numTrees, Array(2, 5, 7, 10 , 15))
      .addGrid(rf.maxDepth ,Array(2, 5, 7, 10 , 15) )
      .addGrid(rf.maxBins, Array(10, 50, 100, 500, 1000))
      .build()
    val r = scala.util.Random
    paramGrid = r.shuffle(paramGrid.toList).toArray

    // 3- Evaluator
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    // Create Hyperband & Train Validation Split objects
    //==================================================================================================================

    // 1- Hyperband
    val hb = new Hyperband()
      .setEstimator(rf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setEta(3)
      .setmaxResource(100)
      .setLogFilePath("/home/eissa/debug.txt")
      .setLogToFile(false)
      .setCollectSubModels(false)

    // 2- Train Validation Split
    val tvs = new TrainValidationSplit()
      .setEstimator(rf)
      .setEvaluator(evaluator)
      .setEstimatorParamMaps(paramGrid)
      .setCollectSubModels(false)


    // Run Hyperband & Train Validation Split Algorithm
    //==================================================================================================================

    val pwLog1 = new PrintWriter(new File("/home/eissa/result.txt" ))
    pwLog1.write("--------------\n")


    // 1- Run Hyperband, and choose the best set of parameters.
    val s1 =  new java.util.Date().getTime
    val model1 = hb.fit(trainingData)
    val e1 =  new java.util.Date().getTime
    val p1 = e1 - s1
    pwLog1.write("Hyperband :\n")
    pwLog1.write("Elapsed Time (Second):" + p1 / 1000.0 + "\n")
    pwLog1.write("Best Model Validation Mrtrics:" + model1.validationMetrics(0) + "\n")
    pwLog1.write("Best Model Hyper-Parameters:" + model1.bestModel.extractParamMap() + "\n")
    pwLog1.write("======================================================================\n")


    // 2- Run train validation split, and choose the best set of parameters.
    val s2 =  new java.util.Date().getTime
    val model2 = tvs.fit(trainingData)
    val e2 =  new java.util.Date().getTime
    val p2 = e2 - s2
    pwLog1.write("train validation split :\n")
    pwLog1.write("Elapsed Time (Second):" + p2 / 1000.0 + "\n")
    pwLog1.write("Best Model Validation Mrtrics:" + model2.validationMetrics.max + "\n")
    pwLog1.write("Best Model Hyper-Parameters:" + model2.bestModel.extractParamMap() + "\n")
    pwLog1.close()
    // Make predictions on test data. model is the model with combination of parameters
    // that performed best.
    //model.bestModel.transform(testData)
    //  .select("features", label, "prediction")
    //  .show()



  }


}
