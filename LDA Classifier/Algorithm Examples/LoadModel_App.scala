import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}


/**
  * Example of loading and saving the model to the hard disk
  */
object LoadModel_App {

  def main(args: Array[String]): Unit = {

    var dataFolderPath = "/home/eissa/mycode/data/"

    //Create Spark Session
    //==================================================================================================================
    val spark = SparkSession
      .builder()
      .appName("Java Spark SQL basic example")
      .config("spark.master", "local")
      .getOrCreate();

    //Stop Information except error
    //==================================================================================================================
    spark.sparkContext.setLogLevel("ERROR")


    //Example 1 (iris) [QDA:0.9473684210526315]
    //=====================================
    val rawdata = spark.read.option("header", "true")
      .option("inferSchema", "true")
      .format("csv")
      .load(dataFolderPath + "iris.data.csv")
    var label = "Class"

    // Example 2 (Sensor) [QDA:0.8570199587061252]
    //======================================
    //val rawdata = spark.read.option("header","true")
    //              .option("inferSchema","true")
    //             .format("csv")
    //             .load(dataFolderPath + "Sensorless_drive_diagnosis.csv")
    //var label = "y"


    //Example 3 (credit_card_clients) [QDA:0.600267156453498]
    //======================================
    //val rawdata = spark.read.option("header", "true")
    //              .option("inferSchema", "true")
    //               .format("csv")
    //               .load(dataFolderPath + "credit_card_clients.csv")
    //var label = "Y"


    //Example 4 (SUSY)
    //======================================
    //val rawdata = spark.read.option("header", "false")
    //             .option("inferSchema", "true")
    //              .format("csv")
    //              .load(dataFolderPath  + "SUSY.csv")
    //             .toDF("Class" , "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18")
    //var label = "Class"


    //Example 5 (Statlog_Shuttle) [DQA: 0.91988981980948]
    //======================================
    //val rawdata = spark.read.option("header", "false")
    //             .option("inferSchema", "true")
    //              .option("delimiter" , ",")
    //             .format("csv")
    //              .load(dataFolderPath  + "Statlog_Shuttle.csv")
    //            .toDF( "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "Class").filter( "Class < 5")
    //var label = "Class"


    //Example 6 avila.txt
    //======================================
    //val rawdata = spark.read.option("header", "false")
    //             .option("inferSchema", "true")
    //              .option("delimiter" , ",")
    //              .format("csv")
    //              .load(dataFolderPath  + "avila.txt")
    //             .toDF( "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10", "Class")
    //var label = "Class"


    //Process data (create vector assembler + scale data)
    //==================================================================================================================
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

    //train algorithm and test it
    //==================================================================================================================

    var outputmodel = org.apache.spark.ml.classification.LDAModel.load(dataFolderPath + "saved")

    var result = outputmodel.transform(mydataset)
    result = result.select(label , "Prediction")

    import spark.sqlContext.implicits._
    val predictionAndLabels = result.map( r=> ( r.getInt(0).toDouble, r.getDouble(1)) ).rdd

    //get convision matrix
    val metrics = new MulticlassMetrics(predictionAndLabels)
    println("Confusion matrix:")
    println(metrics.confusionMatrix)

    // Overall Statistics
    val accuracy = metrics.accuracy
    println("Summary Statistics")
    println(s"Accuracy = $accuracy")

    // Precision by label
    val labels = metrics.labels
    labels.foreach { l =>
      println(s"Precision($l) = " + metrics.precision(l))
    }
    //==================================================================================================================

  }
}
