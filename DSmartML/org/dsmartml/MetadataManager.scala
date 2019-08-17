//package org.apache.spark.ml.tuning
package org.dsmartml

import org.apache.spark.ml.classification._
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{MinMaxScaler, StandardScaler, VectorAssembler}
import org.apache.spark.sql._
import org.apache.spark.sql.functions.{col, _}
import org.apache.spark.sql.types.DoubleType

import scala.collection.immutable.ListMap

/**
  * Class Name: MetadataManager
  * Description: this calss responable for Managing the metadata of a given dataset
  *            :  - It extract metadata for a given dataset
  *            :  - It extract statistical metadata for a given dataset
  *            :  - It extract Classification metadata for a given dataset
  * @constructor Create a MetadataManager object allow us to manage (extract) our dataset metadata (
  * @param spark the used spark session
  * @param logger Object of the logger class (used to log time, exception and result)
  * @param TargetCol the dataset label column name (default = y)
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/3/2019
*/
class MetadataManager (spark:SparkSession,logger: Logger, TargetCol:String = "y") {

  /**
    * this function Extract statistical and calssification metadata
    * @param rawdata the input dataset that we need to extract its  metadata
    * @return filled Metadata object
    */
  def ExtractMetadata(rawdata:DataFrame ): DatasetMetadata =  {
    var metadata:DatasetMetadata = ExtractStatisticalMetadata(rawdata )
    metadata = ExtractClassificationMetadata(rawdata , metadata)
    return metadata
  }

  /**
    * this function Extract Statistical Metadata from dataset
    * @param rawdata the input dataset that we need to extract its  metadata
    * @return Metadata object filled with statistical metadata
    */
  def ExtractStatisticalMetadata(rawdata:DataFrame ): DatasetMetadata = {

    println("1 - Extract Statistical Metadata")
    var metadata: DatasetMetadata = new DatasetMetadata()
    val starttime  =  new java.util.Date().getTime




    val starttime1 =  new java.util.Date().getTime
    // # Drop raw if all of its values are missing
    var df = rawdata.na.drop(1)

    //1- number of instances
    metadata.nr_instances = df.count()

    //2- log of umber of instances
    metadata.log_nr_instances = math.log(metadata.nr_instances)

    //3- Number of Features
    var features: Array[String] = df.columns.filter(c => c != TargetCol)
    metadata.nr_features = features.length

    //4- log number of Features
    metadata.log_nr_features = math.log(metadata.nr_features)
    val Endtime1 =  new java.util.Date().getTime
    println("    - Count instances:" + (Endtime1 - starttime1) )
    //5- Features Statistics
    //====================================================================================================
    val MissingValueCountt3 =  new java.util.Date().getTime
    var cond : Column = null
    var currcol: Array[String] = null
    var columncounter = 0
    val iterationcolumns = 200

    // Distinct Value
    var ColumnsDistinctValMap = Map[String, Long]()
    var DistinctValueRow : Row = null
    // sum Value
    var ColumnsSumValMap = Map[String, Double]()
    var SumValueRow : Row = null
    // skewness Value
    var ColumnsskewnValMap = Map[String, Double]()
    var skewnValueRow : Row = null
    // kurtosis Value
    var ColumnskurtValMap = Map[String, Double]()
    var kurtValueRow : Row = null
    // Min Value
    var ColumnMinValMap = Map[String, Double]()
    var MinValueRow : Row = null


    var l = metadata.nr_features / iterationcolumns

    for (c <- 0 to l)
    {

      currcol = features.slice(c * iterationcolumns, (c * iterationcolumns) + (iterationcolumns))
      if (currcol.length > 0 ) {
        // missing values
        cond = currcol.map(x => col(x).isNull).reduce(_ || _)
        //Distinct count
        DistinctValueRow = df.select(currcol.map(c => countDistinct(col(c)).alias(c)): _*).collect()(0)

        //skewn
        skewnValueRow = df.select(currcol.map(c => skewness(col(c)).alias(c)): _*).collect()(0)
        //skewn
        SumValueRow = df.select(currcol.map(c => sum(col(c)).alias(c)): _*).collect()(0)
        //Min
        MinValueRow = df.select(currcol.map(c => min(col(c)).alias(c)): _*).collect()(0)
        //kurt
        kurtValueRow = df.select(currcol.map(c => kurtosis(col(c)).alias(c)): _*).collect()(0)
        metadata.missing_val = metadata.missing_val + df.filter(cond).count()
        for (cc <- currcol) {

          ColumnsDistinctValMap += (cc -> DistinctValueRow(columncounter).asInstanceOf[Long])
          ColumnskurtValMap += (cc -> kurtValueRow(columncounter).asInstanceOf[Number].doubleValue())
          ColumnsskewnValMap += (cc -> skewnValueRow(columncounter).asInstanceOf[Number].doubleValue())
          ColumnsSumValMap += (cc -> SumValueRow(columncounter).asInstanceOf[Number].doubleValue())
          ColumnMinValMap += (cc -> MinValueRow(columncounter).asInstanceOf[Number].doubleValue())

          columncounter = columncounter + 1
        }
        columncounter = 0
        //println(" 100 Features Loop Number:" + c)
      }

    }


    if (ColumnMinValMap.values.toArray.filter(d => d < 0).length > 0)
      metadata.hasNegativeFeatures = true

    val MissingValueCountt4 =  new java.util.Date().getTime
    println("    - Count Missing Values, Kur, skew, min:" + (MissingValueCountt4 - MissingValueCountt3) )
    logger.logTime("Categorical & Continuse Features Statistics:" + (MissingValueCountt4 - MissingValueCountt3) + ",")


    //6- Ratio of missing value
    metadata.ratio_missing_val = metadata.missing_val.toDouble / metadata.nr_instances.toDouble
    val Endtime2 =  new java.util.Date().getTime

    //7 - Number of Numerical Features & Categorical Features
    // here i will remove coulmn with constant value
    // will remove column with non numeric data type
    // accepted type are : ByteType, DecimalType, DoubleType, FloatType, IntegerType, LongType, ShortType
    val CountFeaturesTypet1 =  new java.util.Date().getTime
    import scala.collection.mutable.ListBuffer
    var numerical = new ListBuffer[String]()
    var categorical = new ListBuffer[String]()
    var ColumnsTypes = Map[String, org.apache.spark.sql.types.DataType]()

    for ((k,v) <- ColumnsDistinctValMap)
    {
      if (k != TargetCol) {
        if (v < 2)
          df = df.drop(k)
        else if (v < metadata.log_nr_instances) {
          categorical += k
          // categoricalFeaturesInfo.put(i, v.toInt)
          //if (v > maxcat)
          //  maxcat = v.toInt
        }
        else
          numerical += k
      }
    }
    metadata.nr_numerical_features = numerical.length
    metadata.nr_categorical_features = categorical.length
    val Endtime3 =  new java.util.Date().getTime
    val CountFeaturesTypet2 =  new java.util.Date().getTime
    logger.logTime("Count Categorical & Continuse Features:" + (CountFeaturesTypet2 - CountFeaturesTypet1) + ",")
    println("    - Count Categorical & Continuse Features:" + (CountFeaturesTypet2 - CountFeaturesTypet1) + ",")

    //10  Ratio of Categorical to Numerical Features - DONE
    metadata.ratio_num_cat = 999999999.0
    if (metadata.nr_numerical_features > 0)
      metadata.ratio_num_cat = metadata.nr_categorical_features.toDouble / metadata.nr_numerical_features.toDouble


    // =====> Classes  Statistics
    // ===================================================================================================
    val ClassesStatt1 =  new java.util.Date().getTime
    //11- Class Entropy - DONE
    var prob_classes = new ListBuffer[Double]()
    metadata.class_entropy = 0.0
    //import spark.implicits._
    //var classes = df.select(TargetCol).distinct().map( r => r.getAs[Integer](0)).collect()
    //var classes_prob_Map = df.groupBy(TargetCol).count().map(r => (r.getInt(0), r.getLong(1))).collect()
    var classes_prob_List = df.groupBy(TargetCol).count().collect().toList

    for (x <- classes_prob_List) {
      var prob = (x(1).asInstanceOf[Long].toDouble / metadata.nr_instances)

      prob_classes += prob
      metadata.class_entropy = metadata.class_entropy - prob * math.log(prob)
    }
    metadata.nr_classes = prob_classes.length

    //12 - Maximum Class probability - DONE
    metadata.max_prob = prob_classes.max

    //13- Minimum Class probability - DONE
    metadata.min_prob = prob_classes.min

    //14-  Mean Class probability - DONE
    metadata.mean_prob = prob_classes.sum / prob_classes.length

    //15 -  Standard Deviation of Class probability - DONE
    metadata.std_dev = 0.0
    for (x <- classes_prob_List) {
      var prob = (x(1).asInstanceOf[Long].toDouble / metadata.nr_instances)
      metadata.std_dev = metadata.std_dev + ((prob - metadata.mean_prob) * (prob - metadata.mean_prob))
    }
    metadata.std_dev = metadata.std_dev / (prob_classes.length - 1)
    metadata.std_dev = math.pow(metadata.std_dev, 0.5)

    // 16 -  Dataset Ratio - DONE
    metadata.dataset_ratio = metadata.nr_features.toDouble / metadata.nr_instances.toDouble
    val ClassesStatt2 =  new java.util.Date().getTime
    logger.logTime("Classes Entropy & Statistics:" + (ClassesStatt2 - ClassesStatt1) + ",")
    println("    - Classes Entropy & Statistics:" + (ClassesStatt2 - ClassesStatt1) + ",")


    // =====> Categorical Features Statistics
    // ===================================================================================================
    // ByteType, DecimalType, DoubleType, FloatType, IntegerType, LongType, ShortType
    val CategoricalFeatureStatt1 =  new java.util.Date().getTime
    var symbols = new ListBuffer[Double]()

    if (categorical.length > 0) {

      //17- Symbols Sum - DONE
      metadata.symbols_sum  = ColumnsSumValMap.filterKeys( k => categorical.contains(k)).values.sum

      //18- Symbols Mean - DONE
      metadata.symbols_mean = symbols.sum / categorical.length

      //19- Symbols Standard Deviation - DONE
      metadata.symbols_std_dev = math.pow(symbols.map(i => math.pow((i - metadata.symbols_mean), 2)).sum / categorical.length, 0.5)

      val CategoricalFeatureStatt2 =  new java.util.Date().getTime
      logger.logTime("Categorical Features Statistics:" + (CategoricalFeatureStatt2 - CategoricalFeatureStatt1) + ",")
      println("    - Categorical Features Statistics:" + (CategoricalFeatureStatt2 - CategoricalFeatureStatt1) + ",")


    }


    // Numerical Features Statistics
    //===========================================================================
    val NumericalFeatureStatt1 =  new java.util.Date().getTime
    var skewness_values = new Array[Double](numerical.length)
    var kurtosis_values = new Array[Double](numerical.length)
    var counter = 0


    skewness_values = ColumnsskewnValMap.filterKeys( k => numerical.contains(k)).values.toArray
    kurtosis_values = ColumnskurtValMap.filterKeys( k => numerical.contains(k)).values.toArray

    if (numerical.length > 0) {
      //20. Skewness Minimum - DONE
      metadata.skew_min = skewness_values.min

      //21. Skewness Maximum - DONE
      metadata.skew_max = skewness_values.max

      //22. Skewness Mean - DONE
      metadata.skew_mean = skewness_values.sum / skewness_values.length

      // 23. Skewness Standard deviation - DONE
      metadata.skew_std_dev = math.pow(skewness_values.map(i => math.pow((i - metadata.skew_mean), 2)).sum / skewness_values.length, 0.5)

      //24. Kurtosis Minimum - DONE
      metadata.kurtosis_min = kurtosis_values.min

      //25. Kurtosis Maximum - DONE
      metadata.kurtosis_max = kurtosis_values.max

      // 26. Kurtosis Mean - DONE
      metadata.kurtosis_mean = kurtosis_values.sum / kurtosis_values.length

      // 27. Kurtosis Standard Deviation - DONE
      metadata.kurtosis_std_dev = math.pow(kurtosis_values.map(i => math.pow((i - metadata.kurtosis_mean), 2)).sum / kurtosis_values.length, 0.5)

      val NumericalFeatureStatt2 =  new java.util.Date().getTime
      logger.logTime("Numerical Feature Statistics:" + (NumericalFeatureStatt2 - NumericalFeatureStatt1) + ",")
      println("    -Numerical Feature Statistics:" + (NumericalFeatureStatt2 - NumericalFeatureStatt1) + ",")
    }

    val Endtime =  new java.util.Date().getTime
    val TotalTime = Endtime - starttime
    logger.logTime("Meta Data Extraction Time:" + TotalTime.toString + ",")

    println("   -- Meta Data Extraction Time:" + (TotalTime/1000.0).toString )
    println("   -- Number of Instances:" +  metadata.nr_instances)
    println("   -- Number of Features :" + metadata.nr_features)
    println("   -- Number of Classess :" + metadata.nr_classes)
    println("   -- Number of Categorical Features: :" + metadata.nr_categorical_features)
    println("   -- Number of Numerical Features: :" + metadata.nr_numerical_features)
    println("   -- Number of Missing Values: :" + metadata.missing_val)
    println("--------------------------------------------------------------------------------")

    //_metadata = metadata
    return metadata

  }

  /**
    * this function Extract Classification Metadata from dataset
    * @param rawdata the input dataset that we need to extract its  metadata
    * @param metadata the dataset statistical metadata
    * @return Metadata object filled with metadata
    */
  def ExtractClassificationMetadata(rawdata:DataFrame , metadata: DatasetMetadata  ): DatasetMetadata = {

    // # Drop raw if all of its values are missing
    var df = rawdata.na.drop(1)

    var label = TargetCol
    var featurecolumns = df.columns.filter(c => c != label)

    // handle null by replace it with median
    //=========================================================================
    var missingvalueTime: Long = 0
    if (metadata.missing_val > 0) {
      val missingvaluet1 =  new java.util.Date().getTime
      df = df.select(df.columns.map(c => col(c).cast(DoubleType)): _*)
      // handle null
      val imputer = new org.apache.spark.ml.feature.Imputer()
        .setInputCols(featurecolumns)
        .setOutputCols(featurecolumns.map(c => s"${c}_imputed"))
        .setStrategy("median")

      df = imputer.fit(df).transform(df)
      df = df.select(df.columns.filter(colName => !featurecolumns.contains(colName)).map(colName => col(colName)): _*)
      val missingvaluet2 =  new java.util.Date().getTime
      missingvalueTime = missingvaluet2 - missingvaluet2
    }
    featurecolumns = df.columns.filter(c => c != label)
    // vector assembler
    val indexingt1 =  new java.util.Date().getTime
    val assembler = new VectorAssembler()
      .setInputCols(featurecolumns)
      .setOutputCol("features")
    val mydataset = assembler.transform(df).select(label, "features")
    val indexingt2 =  new java.util.Date().getTime
    logger.logTime("Indexing Data:" + (indexingt2 - indexingt1) + ",")

    // standard scalar
    val scallingt1 =  new java.util.Date().getTime
    val scalerStandard = new StandardScaler()
      .setInputCol("features")
      .setOutputCol("features_Standarized")
      .setWithStd(true)
      .setWithMean(true)
    val scalerStandardModel = scalerStandard.fit(mydataset)
    val df_MeanStdScaled = scalerStandardModel.transform(mydataset).select(label, "features_Standarized")
    val Array(trainingData_MeanStdScaled, testData_MeanStdScaled) = df_MeanStdScaled.randomSplit(Array(0.8, 0.2))

    // Min Scalar
    val scalerMinMax = new MinMaxScaler()
      .setInputCol("features")
      .setOutputCol("features_Standarized")
    val scalerMinMaxModel = scalerMinMax.fit(mydataset)
    val df_MinMaxScaled = scalerMinMaxModel.transform(mydataset).select(label, "features_Standarized")
    val Array(trainingData_MinMaxScaled, testData_MinMaxScaled) = df_MinMaxScaled.randomSplit(Array(0.8, 0.2))
    val scallingt2 =  new java.util.Date().getTime
    logger.logTime("Scalling Data:" + (scallingt2 - scallingt1) + ",")


    // evaluator
    val evaluator = new MulticlassClassificationEvaluator()
      .setLabelCol(label)
      .setPredictionCol("prediction")
      .setMetricName("accuracy")

    logger.logTime("Handling Missing Value:" + missingvalueTime + ",")
    println("Handling Missing Value:" + missingvalueTime + ",")


    // 1- RandomForest   =================================
    var rftime: Long = 0
    val rft1 = new java.util.Date().getTime
    val rf = new RandomForestClassifier()
      .setLabelCol(label)
      .setFeaturesCol("features_Standarized")
      .setMaxBins(1000)
    try {

      val model_rf = rf.fit(trainingData_MeanStdScaled)
      val predictions_rf = model_rf.transform(testData_MeanStdScaled)
      val accuracy_rf = evaluator.evaluate(predictions_rf)
      metadata.accuracyMap += ("RandomForestClassifier" -> accuracy_rf)
      val rft2 = new java.util.Date().getTime
      rftime = rft2 - rft1
    } catch {
      case ex: Exception => metadata.accuracyMap += ("RandomForestClassifier" -> -1)
        logger.logException(metadata.datasetname + " => Random Forest:"+ ex.getMessage + "\n")
        println(ex.getMessage)
        println("**********************************************************")
    }
    logger.logTime("Random Forest Time:" + rftime + ",")
    println("Random Forest Time:" + rftime + ",")

    // 2- Logistic Regression   =================================
    var lrtime: Long = 0
    val lrt1 = new java.util.Date().getTime
    val lr = new LogisticRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)
      .setLabelCol(label)
      .setFeaturesCol("features_Standarized")
    try {
      val model_lr = lr.fit(trainingData_MeanStdScaled)
      val predictions_lr = model_lr.transform(testData_MeanStdScaled)
      val accuracy_lr = evaluator.evaluate(predictions_lr)
      metadata.accuracyMap += ("LogisticRegression" -> accuracy_lr)
      val lrt2 = new java.util.Date().getTime
      lrtime = lrt2 - lrt1
    } catch {
      case ex: Exception => metadata.accuracyMap += ("LogisticRegression" -> -1)
        println(ex.getMessage)
        logger.logException(metadata.datasetname + " => Logistic Regression:"+ ex.getMessage + "\n")
        println("**********************************************************")
    }
    logger.logTime("Logistic Regression Time:" + lrtime + ",")
    println("Logistic Regression Time:" + lrtime + ",")

    // 3- Decision Tree   ======================================
    var dttime: Long = 0
    val dtt1 = new java.util.Date().getTime
    val dt = new DecisionTreeClassifier()
      .setLabelCol(label)
      .setFeaturesCol("features_Standarized")
      .setMaxBins(1000)

    try {
      val model_dt = dt.fit(trainingData_MeanStdScaled)
      val predictions_dt = model_dt.transform(testData_MeanStdScaled)
      val accuracy_dt = evaluator.evaluate(predictions_dt)
      metadata.accuracyMap += ("DecisionTreeClassifier" -> accuracy_dt)
      val dtt2 = new java.util.Date().getTime
      dttime = dtt2 - dtt1
    } catch {
      case ex: Exception => metadata.accuracyMap += ("DecisionTreeClassifier" -> -1)
        println(ex.getMessage)
        logger.logException(metadata.datasetname + " => Decision Tree:"+ ex.getMessage + "\n")
        println("**********************************************************")
    }
    logger.logTime("Decision Tree Time:" + dttime + ",")
    println("Decision Tree Time:" + dttime + ",")

    // 4- Multilayer Perceptron    ============================
    var mprtime: Long = 0
    val mprt1 = new java.util.Date().getTime
    val layers = Array[Int](featurecolumns.length, 3, metadata.nr_classes)
    val mpr = new MultilayerPerceptronClassifier()
      .setLayers(layers)
      .setBlockSize(128)
      .setSeed(1234L)
      .setMaxIter(100)
      .setLabelCol(label)
      .setFeaturesCol("features_Standarized")
    try {
      val model_mpr = mpr.fit(trainingData_MeanStdScaled)
      val predictions_mpr = model_mpr.transform(testData_MeanStdScaled)
      val accuracy_mpr = evaluator.evaluate(predictions_mpr)
      metadata.accuracyMap += ("MultilayerPerceptronClassifier" -> accuracy_mpr)
      val mprt2 = new java.util.Date().getTime
      mprtime = mprt2 - mprt1
    } catch {
      case e: Exception => metadata.accuracyMap += ("MultilayerPerceptronClassifier" -> -1)
        println(e.getMessage)
        logger.logException(metadata.datasetname + " => Multilayer Perceptron:"+ e.getMessage + "\n")
        println("**********************************************************")
    }
    logger.logTime("Multilayer Perceptron Time:" + mprtime + ",")
    println("Multilayer Perceptron Time:" + mprtime + ",")


    // 5- Linear SVC   ===================================
    var lsvctime: Long = 0
    val lsvct1 = new java.util.Date().getTime
    if (metadata.nr_classes == 2) {
      val lsvc = new LinearSVC()
        .setMaxIter(10)
        .setRegParam(0.1)
        .setLabelCol(label)
        .setFeaturesCol("features_Standarized")
      try {
        val model_lsvc = lsvc.fit(trainingData_MeanStdScaled)
        val predictions_lsvc = model_lsvc.transform(testData_MeanStdScaled)
        val accuracy_lsvc = evaluator.evaluate(predictions_lsvc)
        metadata.accuracyMap += ("LinearSVC" -> accuracy_lsvc)
        val lsvct2 = new java.util.Date().getTime
        lsvctime = lsvct2 - lsvct1
      } catch {
        case ex: Exception => metadata.accuracyMap += ("LinearSVC" -> -1)
          println(ex.getMessage)
          logger.logException(metadata.datasetname + " => Linear SVC:"+ ex.getMessage + "\n")
          println("**********************************************************")
      }
    }
    else
      metadata.accuracyMap += ("LinearSVC" -> 0.0)
    logger.logTime("Linear SVC  Time:" + lsvctime + ",")
    println("Linear SVC  Time:" + lsvctime + ",")

    // 6- NaiveBayes   =================================
    var nbtime: Long = 0
    val nbt1 = new java.util.Date().getTime
    if (metadata.hasNegativeFeatures == false || metadata.hasNegativeFeatures == true ) {
      val nb = new NaiveBayes()
        .setLabelCol(label)
        .setFeaturesCol("features_Standarized")

      try {
        val model_nb = nb.fit(trainingData_MinMaxScaled)
        val predictions_nb = model_nb.transform(testData_MinMaxScaled)
        val accuracy_nb = evaluator.evaluate(predictions_nb)
        metadata.accuracyMap += ("NaiveBayes" -> accuracy_nb)
        val nbt2 = new java.util.Date().getTime
        nbtime = nbt2 - nbt1
      } catch {
        case ex: Exception => metadata.accuracyMap += ("NaiveBayes" -> -1)
          println(ex.getMessage)
          logger.logException(metadata.datasetname + " => NaiveBayes:"+ ex.getMessage + "\n")
          println("**********************************************************")
      }
    }
    else
      metadata.accuracyMap += ("NaiveBayes" -> 0.0)
    logger.logTime("NaiveBayes Time:" + nbtime + ",")
    println("NaiveBayes Time:" + nbtime + ",")


    // 7- GBT   ========================================================
    var gbttime: Long = 0
    val gbtt1 = new java.util.Date().getTime

    if (metadata.nr_classes == 2) {
      val gbt = new GBTClassifier()
        .setLabelCol(label)
        .setFeaturesCol("features_Standarized")
        .setMaxIter(10)
        .setFeatureSubsetStrategy("auto")
        .setMaxBins(1000)
      try {
        val model_gbt = gbt.fit(trainingData_MeanStdScaled)
        val predictions_gbt = model_gbt.transform(testData_MeanStdScaled)
        val accuracy_gbt = evaluator.evaluate(predictions_gbt)
        metadata.accuracyMap += ("GBTClassifier" -> accuracy_gbt)
        val gbtt2 = new java.util.Date().getTime
        gbttime = gbtt2 - gbtt1
      } catch {
        case ex: Exception => metadata.accuracyMap += ("GBTClassifier" -> -1)
          println(ex.getMessage)
          logger.logException(metadata.datasetname + " => GBT:"+ ex.getMessage + "\n")
          println("**********************************************************")
      }
    }
    else
      metadata.accuracyMap += ("GBTClassifier" -> 0.0)
    logger.logTime("GBT Time:" + gbttime + ",")
    println("GBT Time:" + gbttime + ",")

    // 8- LDA   ==================================================
    var ldatime: Long = 0
    val ldat1 = new java.util.Date().getTime
    val lda = new org.apache.spark.ml.classification.LDA()
    //val lda = new org.apache.spark.ml.classification.LDA()
    lda.sc = spark.sparkContext
    lda.setLabelCol(label)
    lda.setFeaturesCol("features_Standarized")
    lda.setScaledData(false)
    lda.setPredictionCol("prediction")
    lda.setProbabilityCol("Probability")
    lda.setRawPredictionCol("RawPrediction")

    try {
      val model_lda = lda.fit(trainingData_MeanStdScaled)
      val predictions_lda = model_lda.transform(testData_MeanStdScaled)
      val accuracy_lda = evaluator.evaluate(predictions_lda)
      metadata.accuracyMap += ("LDA" -> accuracy_lda)
      val ldat2 = new java.util.Date().getTime
      ldatime = ldat2 - ldat1
    } catch {
      case ex: Exception => metadata.accuracyMap += ("LDA" -> -1)
        println(ex.getMessage)
        logger.logException(metadata.datasetname + " => LDA:"+ ex.getMessage + "\n")
        println("**********************************************************")
    }
    logger.logTime("LDA Time:" + ldatime + ",")
    println("LDA Time:" + ldatime + ",")


    // 9- QDA  ===========================================================
    var qdatime: Long = 0
    val qdat1 = new java.util.Date().getTime
    val qda = new org.apache.spark.ml.classification.QDA(spark.sparkContext)
    qda.setLabelCol(label)
    qda.setFeaturesCol("features_Standarized")
    qda.setScaledData(false)
    qda.setPredictionCol("prediction")
    qda.setProbabilityCol("Probability")
    qda.setRawPredictionCol("RawPrediction")
    try {
      val model_qda = qda.fit(trainingData_MeanStdScaled)
      val predictions_qda = model_qda.transform(testData_MeanStdScaled)
      val accuracy_qda = evaluator.evaluate(predictions_qda)
      metadata.accuracyMap += ("QDA" -> accuracy_qda)
      val qdat2 = new java.util.Date().getTime
      qdatime = qdat2 - qdat1
    } catch {
      case ex: Exception => metadata.accuracyMap += ("QDA" -> -1)
        println(ex.getMessage)
        logger.logException(metadata.datasetname + " => QDA:"+ ex.getMessage + "\n")
        println("**********************************************************")
    }
    logger.logTime("QDA Time:" + qdatime + ",")
    println("QDA Time:" + qdatime + ",")

    for( i <- 0 to 8)
      metadata.accOrderMap += ( ListMap(metadata.accuracyMap.toSeq.sortWith(_._2 > _._2):_*).keys.toList(i) -> (i+1))

    metadata.RandomForestClassifier_Accuracy = metadata.accuracyMap("RandomForestClassifier")
    metadata.LogisticRegression_Accuracy =  metadata.accuracyMap("LogisticRegression")
    metadata.DecisionTreeClassifier_Accuracy = metadata.accuracyMap("DecisionTreeClassifier")
    metadata.MultilayerPerceptronClassifier_Accuracy = metadata.accuracyMap("MultilayerPerceptronClassifier")
    metadata.LinearSVC_Accuracy = metadata.accuracyMap("LinearSVC")
    metadata.NaiveBayes_Accuracy = metadata.accuracyMap("NaiveBayes")
    metadata.GBTClassifier_Accuracy = metadata.accuracyMap("GBTClassifier")
    metadata.LDA_Accuracy = metadata.accuracyMap("LDA")
    metadata.QDA_Accuracy = metadata.accuracyMap("QDA")

    metadata.RandomForestClassifier_Order = metadata.accOrderMap("RandomForestClassifier")
    metadata.LogisticRegression_Order =  metadata.accOrderMap("LogisticRegression")
    metadata.DecisionTreeClassifier_Order = metadata.accOrderMap("DecisionTreeClassifier")
    metadata.MultilayerPerceptronClassifier_Order = metadata.accOrderMap("MultilayerPerceptronClassifier")
    metadata.LinearSVC_Order = metadata.accOrderMap("LinearSVC")
    metadata.NaiveBayes_Order = metadata.accOrderMap("NaiveBayes")
    metadata.GBTClassifier_Order = metadata.accOrderMap("GBTClassifier")
    metadata.LDA_Order = metadata.accOrderMap("LDA")
    metadata.QDA_Order = metadata.accOrderMap("QDA")


    metadata.BestAlgorithm_Accuracy = metadata.accuracyMap.maxBy(_._2)._2
    metadata.BestAlgorithm = metadata.accuracyMap.maxBy(_._2)._1

    // Calculate Accuracy Thresholds
    //------------------------------------------------------

    var acc_arr = metadata.accuracyMap.values.filter(x => x > 0 )
    var Acc_Max = acc_arr.max
    var Acc_Mean= acc_arr.sum / acc_arr.size
    var Acc_Std_dev = 0.0

    // Std. Deviation Threshold
    for (x <- acc_arr) {
      Acc_Std_dev = Acc_Std_dev + ((x - Acc_Mean) * (x - Acc_Mean))
    }
    Acc_Std_dev = Acc_Std_dev / (acc_arr.size - 1)
    Acc_Std_dev = math.pow(Acc_Std_dev, 0.5)
    metadata.Acc_Std_threshold = Acc_Max - Acc_Std_dev

    // Order Threshold
    var new_acc_arr = metadata.accuracyMap
    new_acc_arr -=(metadata.accuracyMap.maxBy(_._2)._1)
    for(i <- 0 to metadata.topn-2) {
      new_acc_arr -= (new_acc_arr.maxBy(_._2)._1)
    }
    metadata.Acc_Ord_threshold = new_acc_arr.maxBy(_._2)._2
    //_metadata = metadata
    return metadata

  }


}

object MetadataManager
{
  def GetMetadataSeq( ClassifierIndex: Integer, metadata: DatasetMetadata): Array[Double]=  {


    return Array(
      metadata.nr_instances ,
      metadata.log_nr_instances ,
      metadata.nr_features ,
      metadata.log_nr_features ,
      metadata.nr_classes ,
      metadata.nr_numerical_features ,
      metadata.nr_categorical_features ,
      metadata.ratio_num_cat ,
      metadata.class_entropy ,
      metadata.missing_val ,
      metadata.ratio_missing_val ,
      metadata.max_prob ,
      metadata.min_prob ,
      metadata.mean_prob ,
      metadata.std_dev ,
      metadata.dataset_ratio ,
      metadata.symbols_sum ,
      metadata.symbols_mean ,
      metadata.symbols_std_dev ,
      metadata.skew_min ,
      metadata.skew_max ,
      metadata.skew_mean ,
      metadata.skew_std_dev ,
      metadata.kurtosis_min ,
      metadata.kurtosis_max ,
      metadata.kurtosis_mean ,
      metadata.kurtosis_std_dev,
      ClassifierIndex.toDouble)


  }

}