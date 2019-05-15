import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.SparkSession
import org.apache.spark.ml.feature.{StandardScaler, StandardScalerModel}


object HIGGS_App {

  def main(args: Array[String]): Unit = {

    var dataFolderPath = "wasb:///example/data/"

    //Create Spark Session
    //==================================================================================================================
    val spark = SparkSession
      .builder()
      .appName("HIGGS Classification")
      .config("spark.some.config.option", "some-value")
      .getOrCreate()

    //Stop Information except error
    //==================================================================================================================
    //spark.sparkContext.setLogLevel("ERROR")


    //Example 4 (HIGGS_)
    //======================================
    val rawdata_testing = spark.read.option("header", "false")
      .option("inferSchema", "true")
      .format("csv")
      .load(dataFolderPath + "HIGGS_training_data.csv")
      .toDF(  "c0" , "Label" , "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
       "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18" , "C19", "C20", "C21", "C22", "C23"
        , "C24", "C25", "C26", "C27", "C28")//.repartition(32)

    val rawdata_training = spark.read.option("header", "false")
      .option("inferSchema", "true")
      .format("csv")
      .load(dataFolderPath + "HIGGS_testing_data.csv")
      .toDF("c0" , "Label" , "C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8", "C9", "C10",
        "C11", "C12", "C13", "C14", "C15", "C16", "C17", "C18" , "C19", "C20", "C21", "C22", "C23"
        , "C24", "C25", "C26", "C27", "C28")

    var label = "Label"

    // union the two dataframe and cast data from string to float
    //==================================================================================================================
    var rawdata = rawdata_training.union(rawdata_testing).drop("c0")
    rawdata = rawdata.withColumn("Label", rawdata("Label").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C1", rawdata("C1").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C2", rawdata("C2").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C3", rawdata("C3").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C4", rawdata("C4").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C5", rawdata("C5").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C6", rawdata("C6").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C7", rawdata("C7").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C8", rawdata("C8").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C9", rawdata("C9").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C10", rawdata("C10").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C11", rawdata("C11").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C12", rawdata("C12").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C13", rawdata("C13").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C14", rawdata("C14").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C15", rawdata("C15").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C16", rawdata("C16").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C17", rawdata("C17").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C18", rawdata("C18").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C19", rawdata("C19").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C20", rawdata("C20").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C21", rawdata("C21").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C22", rawdata("C22").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C23", rawdata("C23").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C24", rawdata("C24").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C25", rawdata("C25").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C26", rawdata("C26").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C27", rawdata("C27").cast(org.apache.spark.sql.types.DataTypes.FloatType))
    rawdata = rawdata.withColumn("C28", rawdata("C28").cast(org.apache.spark.sql.types.DataTypes.FloatType))

    rawdata.persist()


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

    val lda = new org.apache.spark.ml.classification.LDA( spark.sparkContext)
    lda.setLabelCol(label)
    lda.setFeaturesCol("features")
    lda.setScaledData(false)
    lda.setPredictionCol("Prediction")
    lda.setProbabilityCol("Probability")
    lda.setRawPredictionCol("RawPrediction")


    val Alldata = assembler.transform(rawdata.na.drop).select(label, "features_in")
    val scalermodel = scaler.fit(Alldata)


   var Array(trainingDS, testingDS) = scalermodel.transform(Alldata).randomSplit(Array(0.8, 0.2))


   //train algorithm and test it
   //==================================================================================================================
   var model = lda.fit(trainingDS)

   var result = model.transform(testingDS)
   result = result.select(label , "Prediction")

   import spark.sqlContext.implicits._
   val predictionAndLabels = result.map( r=> ( r.getFloat(0).toDouble, r.getDouble(1)) ).rdd

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

    rawdata.unpersist()

  }
}
