//package org.apache.spark.ml.tuning
package org.dsmartml

import java.nio.file.StandardCopyOption

import org.apache.spark.ml.Model
import org.apache.spark.ml.classification._
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.ml.linalg.DenseMatrix
import org.apache.spark.mllib.evaluation.MulticlassMetrics
import org.apache.spark.sql.types.{DoubleType, StructField, StructType}
import org.apache.spark.sql.{DataFrame, Row, SparkSession}

import scala.collection.mutable.ArrayBuffer


/**
  * Class Name: KBManager
  * Description: this calss responable for Managing the KnowledgeBase
  *            :  - It can load the KnowledgeBase to Use it
  *            :  - It build & Save Model for select best Classifier
  *            :  - It save dataset metadata (statistics & classification) to KB csv File
  * @constructor Create a KBManager allow us to use our KB
  * @param spark the used spark session
  * @param logger Object of the logger class (used to log time, exception and result)
  * @param TargetCol the dataset label column name (default = y)
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/3/2019
  */
class KBManager(spark:SparkSession, logger:Logger, TargetCol:String = "y") extends java.io.Serializable {

  /**
    * Empty DatasetMetadata object
    */
  var _metadata : DatasetMetadata = null
  /**
    *  Map contains the Algorithm name as key and its (Accuracy) as value
    */
  var KBModelaccuracyMap = Map[String, Double]()
  /**
    *  Map contains the Algorithm name as key and its (Precision) as value
    */
  var KBModelprecisionMap = Map[String, Double]()
  /**
    *  Map contains the Algorithm name as key and its (Recal) as value
    */
  var KBModelrecallMap = Map[String, Double]()
  /**
    *  Map contains the Algorithm name as key and its (F-Score) as value
    */
  var KBModeFmeasureMap = Map[String, Double]()


  /**
    * this function receive Metadata Object as input and write it to our Knowledgebase
    * @param path path of the KB
    * @param metadata Dataset Metadata Object
    */
  def SaveMetadata_toKB (path:String , metadata: DatasetMetadata)=  {

    var metadatastatistics = metadata.datasetname + "," + metadata.nr_instances + "," + metadata.log_nr_instances + "," + metadata.nr_features + "," + metadata.log_nr_features + "," + metadata.nr_classes + "," + metadata.nr_numerical_features + "," + metadata.nr_categorical_features + "," +
      metadata.ratio_num_cat + "," + metadata.class_entropy + "," + metadata.missing_val + "," + metadata.ratio_missing_val + "," + metadata.max_prob + "," + metadata.min_prob + "," + metadata.mean_prob + "," + metadata.std_dev + "," +
      metadata.dataset_ratio + "," + metadata.symbols_sum + "," + metadata.symbols_mean + "," + metadata.symbols_std_dev + "," + metadata.skew_min + "," + metadata.skew_max + "," + metadata.skew_mean + "," + metadata.skew_std_dev + "," +
      metadata.kurtosis_min + "," + metadata.kurtosis_max + "," + metadata.kurtosis_mean + "," + metadata.kurtosis_std_dev + ","

    //Accuracy & Order
    logger.logResult(metadatastatistics + metadata.RandomForestClassifier_Accuracy + ",RandomForestClassifier," + metadata.RandomForestClassifier_Order + "," +  getClassifierLabel(metadata.RandomForestClassifier_Accuracy , metadata.Acc_Std_threshold ,metadata.Acc_Ord_threshold )+ "\n")
    logger.logResult(metadatastatistics + metadata.LogisticRegression_Accuracy     + ",LogisticRegression," + metadata.LogisticRegression_Order + "," + getClassifierLabel(metadata.LogisticRegression_Accuracy , metadata.Acc_Std_threshold ,metadata.Acc_Ord_threshold  )+ "\n")
    logger.logResult(metadatastatistics + metadata.DecisionTreeClassifier_Accuracy + ",DecisionTreeClassifier," + metadata.DecisionTreeClassifier_Order + "," + getClassifierLabel(metadata.DecisionTreeClassifier_Accuracy , metadata.Acc_Std_threshold ,metadata.Acc_Ord_threshold )+ "\n")
    logger.logResult(metadatastatistics + metadata.MultilayerPerceptronClassifier_Accuracy + ",MultilayerPerceptronClassifier," + metadata.MultilayerPerceptronClassifier_Order + "," + getClassifierLabel(metadata.MultilayerPerceptronClassifier_Accuracy , metadata.Acc_Std_threshold ,metadata.Acc_Ord_threshold )+ "\n")
    logger.logResult(metadatastatistics + metadata.LinearSVC_Accuracy + ",LinearSVC," + metadata.LinearSVC_Order + "," + getClassifierLabel(metadata.LinearSVC_Accuracy , metadata.Acc_Std_threshold ,metadata.Acc_Ord_threshold )+ "\n")
    logger.logResult(metadatastatistics + metadata.NaiveBayes_Accuracy + ",NaiveBayes," + metadata.NaiveBayes_Order + "," + getClassifierLabel(metadata.NaiveBayes_Accuracy , metadata.Acc_Std_threshold ,metadata.Acc_Ord_threshold )+ "\n")
    logger.logResult(metadatastatistics + metadata.GBTClassifier_Accuracy + ",GBTClassifier," + metadata.GBTClassifier_Order + "," + getClassifierLabel(metadata.GBTClassifier_Accuracy , metadata.Acc_Std_threshold ,metadata.Acc_Ord_threshold )+ "\n")
    logger.logResult(metadatastatistics + metadata.LDA_Accuracy + ",LDA," + metadata.LDA_Order + "," + getClassifierLabel(metadata.LDA_Accuracy , metadata.Acc_Std_threshold ,metadata.Acc_Ord_threshold )+ "\n")
    logger.logResult(metadatastatistics + metadata.QDA_Accuracy + ",QDA," + metadata.QDA_Order + "," + getClassifierLabel(metadata.QDA_Accuracy , metadata.Acc_Std_threshold ,metadata.Acc_Ord_threshold ) + "\n")
  }


  /**
    * this function Return three comma separted string ex: "Good,Bad,Good" , based on the three threshold: within 1 Std Deviation, top 3 Accuracy, both (max 3 within 1 std deviation)
    * @param Accuracy accuracy of the algorithm
    * @param Acc_Std_threshold threshold for Accurcy (based on standard deviation)
    * @param Acc_Ord_threshold threshold for Accurcy (based on order of the algorithm (top n))
    * @return three comma separted string ex: "Good,Bad,Good"
    */
  def getClassifierLabel( Accuracy: Double, Acc_Std_threshold:Double , Acc_Ord_threshold:Double ): String = {
    var result = ""

    if(Accuracy > Acc_Std_threshold)
      result = result + "1,"
    else
      result = result + "0,"

    if(Accuracy > Acc_Ord_threshold)
      result = result + "1,"
    else
      result = result + "0,"

    if(Accuracy > Acc_Std_threshold && Accuracy > Acc_Ord_threshold )
      result = result +  "1"
    else
      result = result + "0"


    return result

  }

  /**
    * this function Load KB as a Dataframe
    * @return KB as a dataframe
    */
  def LoadKB(): DataFrame =  {
    var rawdata = spark.read.option("header",true)
      .option("inferSchema","true")
      .option("delimiter", ",")
      .format("csv")
      .load( KBManager.KBpath )

    rawdata = rawdata.drop("dataset").drop("Algorithm").drop("Order")
      .drop("accuracy").drop("y2").drop("y3")
    return rawdata
  }

  /**
    * this function Print Knowledgebase Model Metrics (Accuracy, Precision, Recal & F-score)
    * @param Algorithm
    * @param model
    * @param df_test
    */
  def PrintKBClassifierMetric( Algorithm:String,model:Model[_] , df_test: DataFrame): Unit =  {
    val predictions_rf = model.transform(df_test)
    import spark.implicits._
    val output = predictions_rf.select( "prediction" ,KBManager.label  ).map(x=> (x.getDouble(0),x.getInt(1).toDouble)).rdd
    val metrics = new MulticlassMetrics(output)

    val cm = metrics.confusionMatrix.toArray
    val accuracy=(cm(0) + cm(3)) / cm.sum
    val precision=(cm(0))/( cm(0) + cm(1) )
    val recall= (cm(0))/  ( cm(0) + cm(2) )
    val fscore = (2 * ( precision * recall)) / ( precision + recall)

    KBModelaccuracyMap  += (Algorithm -> accuracy)
    KBModelprecisionMap += (Algorithm -> precision)
    KBModelrecallMap    += (Algorithm -> recall)
    KBModeFmeasureMap   += (Algorithm -> fscore)

    //println("---->" + Algorithm)
    //println("accuracy" + accuracy)
    //println("Precision" + precision)
    //println("Recall" + recall)
    //println("F-Score" + fscore)
    //println("==============================================")
  }

  /**
    * this function Create Classisification Model based on the Knowledgebase data, this model will be used to do algorithm selection based on the dataset metadata
    */
  def CreateKBModel(): Unit =  {
    var KBModelaccuracyMap = Map[String, Double]()
    var df = LoadKB()
    var featurecolumns = df.columns.filter(c => c != KBManager.label)
    // vector assembler
    val assembler = new VectorAssembler()
      .setInputCols(featurecolumns)
      .setOutputCol("features")
    val mydataset = assembler.transform(df).select(KBManager.label, "features")
    val Array(trainingData, testData) = mydataset.randomSplit(Array(0.8, 0.2) , 123)


    // 3- GBT Classifier   =================================
    val gbt = new GBTClassifier()
      .setLabelCol(KBManager.label)
      .setFeaturesCol("features")
      .setMaxIter(10)
      .setMaxDepth(6)
      //.setImpurity("entropy")
      .setMaxBins(150)
      .setFeatureSubsetStrategy("all")

    try {
      val model_gbt = gbt.fit(trainingData)
      model_gbt.save(KBManager.KBModelPath)
      val predictions_gbt = model_gbt.transform(testData)
      import spark.implicits._
      val output = predictions_gbt.select( "label" ,"prediction"  ).map(x=> (x.getInt(0).toDouble,x.getDouble(1))).rdd
      val metrics = new MulticlassMetrics(output)
      val cm = metrics.confusionMatrix.toArray

      val accuracy=(cm(0) + cm(3)) / cm.sum
      val precision=(cm(0))/( cm(0) + cm(1) )
      val recall= (cm(0))/  ( cm(0) + cm(2) )
      val fscore = (2 * ( precision * recall)) / ( precision + recall)
      println("GBT Classifier ==============================" )
      println("accuracy" + accuracy)
      println("Precision" + precision)
      println("Recall" + recall)
      println("F-Score" + fscore)
      println("----------------------------------------------------------------")
    } catch
      {
        case ex: Exception => //KBModelaccuracyMap += ("RandomForestClassifier" -> -1)
          logger.logException("KB => GBT:"+ ex.getMessage + "\n")
          println(ex.getMessage)
          println("**********************************************************")
      }
    /*
        // 3-
        // random Classifier   =================================
        val rf = new RandomForestClassifier()
          .setLabelCol(KBManager.label)
          .setFeaturesCol("features")
          .setNumTrees(9)
          .setMaxDepth(8)
          //.setImpurity("entropy")
          //.setMaxBins(10)
          .setFeatureSubsetStrategy("all")

        try {
          val model_rf = rf.fit(trainingData)
          //model_rf.save(KBManager.KBModelPath +"_2" )
          val predictions_rf = model_rf.transform(testData)
          import spark.implicits._
          val output = predictions_rf.select( "label" ,"prediction"  ).map(x=> (x.getInt(0).toDouble,x.getDouble(1))).rdd
          val metrics = new MulticlassMetrics(output)
          val cm = metrics.confusionMatrix.toArray

          val accuracy=(cm(0) + cm(3)) / cm.sum
          val precision=(cm(0))/( cm(0) + cm(1) )
          val recall= (cm(0))/  ( cm(0) + cm(2) )
          val fscore = (2 * ( precision * recall)) / ( precision + recall)
          println("Random Forest Classifier ==============================" )
          println("accuracy" + accuracy)
          println("Precision" + precision)
          println("Recall" + recall)
          println("F-Score" + fscore)
          println("----------------------------------------------------------------")
        } catch
          {
            case ex: Exception => //KBModelaccuracyMap += ("RandomForestClassifier" -> -1)
              logger.logException("KB => Random Forest:"+ ex.getMessage + "\n")
              println(ex.getMessage)
              println("**********************************************************")
          }
    */
  }

  /**
    * this function Load Classisification Model based on the Knowledgebase data
    * @return the Model used to predict any dataset best algorithm based on our KB
    */
  def LoadKBModel(): GBTClassificationModel = {
    CopyModelFromJAR()
    ///media/eissa/New/data/KBModel1
    //val KBModel = GBTClassificationModel.load(KBManager.KBModelPath )
    val KBModel = GBTClassificationModel.load(KBManager.KBModelPath )
    return KBModel
  }

  def CopyModelFromJAR() = {
    try {

      //Read All files as stream
      var is_data_1 = this.getClass().getResourceAsStream("/KBModel/data/_SUCCESS")
      var is_data_2 = this.getClass().getResourceAsStream("/KBModel/data/part-00000-3fb024f1-06fe-453e-9eb1-cb06523920a6-c000.snappy.parquet")
      var is_metadata_1 = this.getClass().getResourceAsStream("/KBModel/metadata/_SUCCESS")
      var is_metadata_2 = this.getClass().getResourceAsStream("/KBModel/metadata/part-00000")
      var is_treesmetadata_1 = this.getClass().getResourceAsStream("/KBModel/treesMetadata/_SUCCESS")
      var is_treesmetadata_2 = this.getClass().getResourceAsStream("/KBModel/treesMetadata/part-00000-35ca87d6-56ac-48f6-b502-af20da817c4e-c000.snappy.parquet")

      //Create folders
      val currentDirectory = new java.io.File(".").getCanonicalPath
      var p_KBModel : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel");
      var p_data : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel/data");
      var p_metadata : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel/metadata");
      var p_treesMetadata : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel/treesMetadata");
      java.nio.file.Files.createDirectories(p_KBModel)
      java.nio.file.Files.createDirectories(p_data)
      java.nio.file.Files.createDirectories(p_metadata)
      java.nio.file.Files.createDirectories(p_treesMetadata)

      //create paths to files
      var p_data_1 : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel/data/_SUCCESS")
      var p_data_2 : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel/data/part-00000-3fb024f1-06fe-453e-9eb1-cb06523920a6-c000.snappy.parquet");
      var p_metadata_1 : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel/metadata/_SUCCESS");
      var p_metadata_2 : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel/metadata/part-00000");
      var p_treesMetadata_1 : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel/treesMetadata/_SUCCESS");
      var p_treesMetadata_2 : java.nio.file.Path = java.nio.file.Paths.get(currentDirectory.toString + "/KBModel/treesMetadata/part-00000-35ca87d6-56ac-48f6-b502-af20da817c4e-c000.snappy.parquet");

      //copy files to new location
      java.nio.file.Files.copy(is_data_1,p_data_1 , StandardCopyOption.REPLACE_EXISTING)
      java.nio.file.Files.copy(is_data_2,p_data_2 , StandardCopyOption.REPLACE_EXISTING)
      java.nio.file.Files.copy(is_metadata_1,p_metadata_1 , StandardCopyOption.REPLACE_EXISTING)
      java.nio.file.Files.copy(is_metadata_2,p_metadata_2 , StandardCopyOption.REPLACE_EXISTING)
      java.nio.file.Files.copy(is_treesmetadata_1,p_treesMetadata_1 , StandardCopyOption.REPLACE_EXISTING)
      java.nio.file.Files.copy(is_treesmetadata_2,p_treesMetadata_2 , StandardCopyOption.REPLACE_EXISTING)

    }
    catch {
      case e:Exception => println("Exception" + e.getMessage)
    }
  }


  /**
    * Predict Good Classification Algorithm based on the KBModel
    * @param rawdata input dataset that we need to select the suitable classifiers for it based on our KB
    * @param ClassifiersListParam use it if you want to limit the number of checked classifiers
    * @return list of the indecies of predict classifiers
    */
  def PredictBestClassifiers(rawdata:DataFrame , ClassifiersListParam:String = ""): List[Int] = {
    //println("Select Best Algorithms based on the KB")
    var metadataMgr = new MetadataManager(spark, logger, TargetCol)
    var classifiersLsit = ClassifiersManager.classifiersLsit
    //var newMetadataSeq = Seq()
    var result:Array[Double] = null
    var numcol = 0

    var metadata = metadataMgr.ExtractStatisticalMetadata(rawdata)

    val starttime1 =  new java.util.Date().getTime

    _metadata = metadata
    if(ClassifiersListParam != "")
      classifiersLsit = classifiersLsit.filter( p =>  ClassifiersListParam.split(",").contains(p))


    for ( i <- classifiersLsit)
    {
      if(result == null) {
        result = MetadataManager.GetMetadataSeq(classifiersLsit.indexOf(i) , metadata)
        numcol = result.length
      }
      else
        result = result ++ (MetadataManager.GetMetadataSeq(classifiersLsit.indexOf(i) , metadata))
    }

    val matrix = new DenseMatrix(classifiersLsit.length,numcol, result , true) //Matrices.dense(classifiersLsit.length,numcol, result )
    val rdd = spark.sparkContext.parallelize(matrix.rowIter.toSeq).map(x => {
      Row.fromSeq(x.toArray.toSeq)
    })
    val sc = spark.sqlContext

    var schema = new StructType()

    var ids = ArrayBuffer[String]()
    for (i <- 0 until matrix.colIter.size) {
      schema = schema.add(StructField("c" +"_"+ i.toString(), DoubleType, true))
      ids.append("c" +"_"+ i.toString())
    }

    val df = sc.sparkSession.createDataFrame(rdd, schema)
    val assembler = new VectorAssembler()
      .setInputCols(df.columns)
      .setOutputCol("features")
    val output = assembler.transform(df)
    import spark.implicits._

    val df_f = output.select( "c_27" , "features")
    var r = LoadKBModel().transform(df_f)
    var selectedclassifiers = r.select("c_27" ).filter("prediction == 1.0").sort("probability").map(x => x.getAs[Double](0)).collect()
    println("2 - Algorithm Selection based on our KB")
    val Endtime1 = new java.util.Date().getTime
    val TotalTime1 = Endtime1 - starttime1
    println("   -- Time:" + (TotalTime1/1000.0).toString )
    var SelectedClassifiersNames = ""
    var res = ClassifiersManager.SortedSelectedClassifiers(selectedclassifiers)
    for ( i <- res) {
      SelectedClassifiersNames = SelectedClassifiersNames + "," + ClassifiersManager.classifiersLsit(i.toInt)
    }
    println("   -- Predicted Best Classifiers:" )
    SelectedClassifiersNames.split(",").foreach( x => if(x.length > 2) println("     - " + x))
    println("------------------------------------------------------------------------")


    return  res


  }


}




object KBManager extends java.io.Serializable{
  // KB csv file path
  var KBpath = "/media/eissa/New/data/KB_Main.csv"
  // KB Label
  var label = "label"
  //KB Saved Model File
  //val KBModelPath = "/media/eissa/New/data/KBModel"
  //val KBModelPath = "wasb:///example/data/KBModel"
  //val KBModelPath = "gs://sparkstorage/KBModel"

  val KBModelPath = new java.io.File(".").getCanonicalPath.toString+ "/KBModel" //this.getClass().getResource("KBModel").getPath// "KBModel"

}
