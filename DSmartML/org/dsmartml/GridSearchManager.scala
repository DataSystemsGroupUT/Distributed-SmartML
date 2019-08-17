//package org.apache.spark.ml.tuning
package org.dsmartml

import org.apache.spark.ml.Model
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.param.ParamMap
import org.apache.spark.ml.tuning.TrainValidationSplit
import org.apache.spark.sql.functions._
import org.apache.spark.sql.{Column, DataFrame, Row, SparkSession}


/**
  * Class Name: GridSearchManager
  * Description: this calss responable for doing Grid search using all avaialbe classifiers & thier hyperparameters values exisit in ClassifiersManager class, the class use TrainValidationSplit class on each algorithm to determine the best accuracy for that algorithm and its hyper-parameters, then it compare between all the algorithms searched and return the best one.
  * @constructor Create an Object of GridSearchManager
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/5/2019
  */
class GridSearchManager {
  /**this function do the Grid search on passed dataframe
    * @param spark the used spark session
    * @param df the dataframe on which we will do the grid search
    * @param TargetCol the label column name
    * @return name of the best algorithm, Object of the trained Mode, the best hyper parameters found and the accuracy
    */
  def Search(spark:SparkSession , df:DataFrame , featureCol:String = "features" , TargetCol:String = "y" ,
             Parallelism:Int = 1, seed:Long=1234 , ConvertToVecAssembly:Boolean = true,
             ScaleData:Boolean = true ):(String, ( Model[_] , ParamMap , Double)) =  {

    var selectedModelMap = Map[String, ( Model[_] , ParamMap , Double)]()


    //prepare dataset by converting to Vector Assembly & Scale it (if needed)
    var mydataset = df
    if(ConvertToVecAssembly)
      mydataset = DataLoader.convertDFtoVecAssembly(df, featureCol,TargetCol )
    if(ScaleData)
      mydataset = DataLoader.ScaleDF(mydataset ,featureCol,TargetCol )

    // get some info about the dataset (number of features, number of classes and if it has negative values)
    val featurecolumns = df.columns.filter(c => c != TargetCol)
    val nr_features:Int = featurecolumns.length
    var nr_classes = df.groupBy(TargetCol).count().collect().toList.length
    val hasNegativeFeatures = HasNegativeValue(df,nr_features,nr_classes,TargetCol)

    //split data
    val Array(trainingData, testData) = mydataset.randomSplit(Array(0.8, 0.2) , seed)

    /*
    val featurecolumns = df.columns.filter(c => c != TargetCol)
    val nr_features:Int = featurecolumns.length
    var nr_classes = df.groupBy(TargetCol).count().collect().toList.length
    val hasNegativeFeatures = HasNegativeValue(df,nr_features,nr_classes,TargetCol)
    val assembler = new VectorAssembler()
      .setInputCols(featurecolumns)
      .setOutputCol("features")

    val mydataset = assembler.transform(df.na.drop).select(TargetCol, "features")
    val Array(trainingData, testData) = mydataset.randomSplit(Array(0.7, 0.3))
    */

    val ClassifierMgr = new ClassifiersManager(spark,nr_features , nr_classes)

    println("Grid Search")
    for (classifier <- ClassifiersManager.classifiersLsit) {
      val i = ClassifiersManager.classifiersLsit.indexOf(classifier)

      if ((nr_classes == 2 && Array(4, 6).contains(i))
          ||
        (nr_classes >= 2 && Array(0, 1, 2, 3).contains(i))
          ||
            (hasNegativeFeatures == false && i == 5)

      ) {

        try {
          println("-- GridSearch for algoritm: " + classifier + " Start")
          val starttime1 = new java.util.Date().getTime
          var tvs = new TrainValidationSplit()
            .setEstimator(ClassifierMgr.ClassifiersMap(classifier))
            .setEvaluator(ClassifierMgr.evaluator)
            .setEstimatorParamMaps(ClassifierMgr.ClassifierParamsMap(classifier))
            .setCollectSubModels(false)
            .setSeed(seed)
            .setParallelism(Parallelism)

          // 2- Run train validation split, and choose the best set of parameters.
          val s2 = new java.util.Date().getTime
          val model2 = tvs.fit(trainingData)
          val e2 = new java.util.Date().getTime
          val p2 = e2 - s2
          val predictions2 = model2.bestModel.transform(testData)
          val accuracy2 = ClassifierMgr.evaluator.evaluate(predictions2)

          selectedModelMap +=
            (classifier -> (model2, model2.bestModel.extractParamMap(), accuracy2))
          val Endtime1 = new java.util.Date().getTime
          val TotalTime1 = Endtime1 - starttime1
          println("   -- GridSearch for algoritm: " + classifier + " End (Time:" + (TotalTime1 / 1000.0).toString + ")  Accuracy: " + accuracy2)
        } catch {
          case ex: Exception =>
            println("   -- Exception (GridSearch - Search - " + classifier + " ):" + ex.getMessage())
            ex.printStackTrace()
        }


      }



      // For LDA
      if (i == 7) {
        val starttime1 = new java.util.Date().getTime
        var classifier = ClassifiersManager.classifiersLsit(i.toInt)
        // evaluator
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol(TargetCol)
          .setPredictionCol("prediction")
          .setMetricName("accuracy")

        val lda = new org.apache.spark.ml.classification.LDA()
        lda.sc = spark.sparkContext
        lda.setLabelCol(TargetCol)
        lda.setFeaturesCol(featureCol)
        lda.setScaledData(false)
        lda.setPredictionCol("prediction")
        lda.setProbabilityCol("Probability")
        lda.setRawPredictionCol("RawPrediction")
        //lda.sc = spark.sparkContext
        val model_lda = lda.fit(trainingData)
        val predictions_lda = model_lda.transform(testData)
        val accuracy_lda = evaluator.evaluate(predictions_lda)
        selectedModelMap += (classifier -> (model_lda, null, accuracy_lda))

        val Endtime1 = new java.util.Date().getTime
        val TotalTime1 = Endtime1 - starttime1
        println("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ") Accuracy: " + accuracy_lda)
      }

      //For QDA
      if (i == 8) {
        val starttime1 = new java.util.Date().getTime
        var classifier = ClassifiersManager.classifiersLsit(i.toInt)
        // evaluator
        val evaluator = new MulticlassClassificationEvaluator()
          .setLabelCol(TargetCol)
          .setPredictionCol("prediction")
          .setMetricName("accuracy")

        val qda = new org.apache.spark.ml.classification.QDA(spark.sparkContext)
        qda.setLabelCol(TargetCol)
        qda.setFeaturesCol(featureCol)
        qda.setScaledData(false)
        qda.setPredictionCol("prediction")
        qda.setProbabilityCol("Probability")
        qda.setRawPredictionCol("RawPrediction")
        val model_qda = qda.fit(trainingData)
        val predictions_qda = model_qda.transform(testData)
        val accuracy_qda = evaluator.evaluate(predictions_qda)
        selectedModelMap += (classifier -> (model_qda, null, accuracy_qda))

        val Endtime1 = new java.util.Date().getTime
        val TotalTime1 = Endtime1 - starttime1
        println("   -- Hyperband for algoritm:" + classifier + " (Time:" + (TotalTime1 / 1000.0).toString + ")  Accuracy: " + accuracy_qda)
      }
    }
    return Map(selectedModelMap.toSeq.sortWith(_._2._3 > _._2._3):_*).head
  }

  /**
    * this function check if the dataset has negative values in any column or not
    * @param df the dataframe that we will do Grid Search on it
    * @param nr_features the number of features in the dataset
    * @param nr_classes the number of classes in the dataset
    * @param TargetCol the label column name
    * @return boolean (true = has negative values, false = no negative values)
    */
  def HasNegativeValue(df:DataFrame , nr_features:Int, nr_classes:Int , TargetCol:String): Boolean =  {
    val iterationcolumns = 1000
    var cond : Column = null
    var columncounter = 0
    // Min Value
    var ColumnMinValMap = Map[String, Double]()
    var MinValueRow : Row = null

    var l = nr_features / iterationcolumns
    var currcol: Array[String] = null
    var features: Array[String] = df.columns.filter(c => c != TargetCol)

    for (c <- 0 to l)
    {
      currcol = features.slice(c * iterationcolumns, (c * iterationcolumns) + (iterationcolumns))
      if (currcol.length > 0 ) {
        // missing values
        cond = currcol.map(x => col(x).isNull).reduce(_ || _)

        //Min
        MinValueRow = df.select(currcol.map(c => min(col(c)).alias(c)): _*).collect()(0)

        for (cc <- currcol) {
          ColumnMinValMap += (cc -> MinValueRow(columncounter).asInstanceOf[Number].doubleValue())
          columncounter = columncounter + 1
        }
        columncounter = 0
      }

    }

    if (ColumnMinValMap.values.toArray.filter(d => d < 0).length > 0)
        return true
    else
      return false

  }
}
