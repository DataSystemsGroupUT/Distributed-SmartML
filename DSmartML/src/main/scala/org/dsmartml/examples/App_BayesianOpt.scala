package org.dsmartml.examples

import breeze.linalg
import breeze.linalg.DenseVector
import org.apache.spark.ml.{Estimator, Model, Pipeline}
import org.apache.spark.ml.classification._
import org.apache.spark.ml.param.{ParamMap, ParamPair}
import org.apache.spark.sql.{DataFrame, Dataset, SparkSession}
import org.apache.spark.sql.functions.col
import org.apache.spark.sql.types.DoubleType
import org.apache.spark.ml.regression.{LinearRegression, RandomForestRegressionModel, RandomForestRegressor}
import org.apache.spark.ml.feature.{StringIndexer, VectorIndexer}
import org.apache.spark.ml.evaluation.RegressionEvaluator
import org.apache.spark.util.ThreadUtils
import org.dsmartml._
import org.apache.spark.sql.functions.desc
import org.dsmartml.knowledgeBase.KBManager

import scala.collection.immutable.ListMap
import scala.collection.mutable.ListBuffer
import scala.concurrent.{Future, TimeoutException}
import scala.util.control.Breaks._

object App_BayesianOpt{

  var paramMapList = new ListBuffer[ParamMap]()
  var ParamMapAccuracyLst = new ListBuffer[(ParamMap, Double) ]()
  var Allparam_df:DataFrame = null
  var logger:Logger = null
  var pm2:ParamMap = new ParamMap()
  var pl = new ListBuffer[ParamMap]()

  //var ParamMapAccuracyLst = new ListBuffer[ParamMap , ]()
  def main(args: Array[String]): Unit = {

    var dataFolderPath = "/media/eissa/New/data/"
    var logpath = "/media/eissa/New/data/"

    // Create Spark Session
    implicit val spark = SparkSession
      .builder()
      .appName("Distributed Smart ML 1.0")
      .config("spark.master", "local")
      .getOrCreate()

    // Set Log Level to error only (don't show warning)
    spark.sparkContext.setLogLevel("ERROR")
    var TargetCol = "y"


    var classifier =  "MultilayerPerceptronClassifier" //LinearSVC" //GBTClassifier DecisionTreeClassifier - LogisticRegression - RandomForestClassifier
    // Initialization ...
    // 1- Create needed objects
        logger = new Logger(logpath)
         var metadataMgr = new MetadataManager(spark, logger, TargetCol)

    // 4- Load Dataset (and work on fraction of it)
    //-------------------------------------------------------------------------------
    var dataloader = new DataLoader(spark, 7, dataFolderPath, logger)
    var rawdata = dataloader.getData()
    var metadata = metadataMgr.ExtractStatisticalMetadataSimple(rawdata)


    var classifiermgr = new ClassifiersManager(spark,metadata.nr_features,metadata.nr_classes)

    // 2- get All Random Parameters for this classifier
        var AllparamMapList = classifiermgr.generateParametersForBayesianOpt(classifier ) //"RandomForestClassifier")

        var AllParamMapAccLst = new ListBuffer[ (ParamMap , Double)]()
        AllparamMapList.foreach( e => AllParamMapAccLst.append( (e,1) ))

        //Allparam_df = convertParamtoDF_withoutAccuracy(spark, AllparamMapList , "RandomForestClassifier")
    Allparam_df = convertParamtoDF(spark, AllParamMapAccLst , true, classifier)
    Allparam_df = DataLoader.convertDFtoVecAssembly_WithoutLabel( Allparam_df , "features" )


    // 3- Get initial Configuration
    //-------------------------------------------------------------------------------
      val count = AllparamMapList.length;
      import scala.util.Random
      val rand = new Random()
      val numberOfElements = 5
      var i = 0
      while ( i < numberOfElements)
      {
        val randomIndex = rand.nextInt(AllparamMapList.size)
        val randomElement = AllparamMapList(randomIndex)
        paramMapList.append(randomElement)
        i += 1
      }

     val Array(trainingDataset_init, validationDataset_init) = rawdata.randomSplit(Array(0.05, 0.95), 1234)
      //val Array(trainingDataset_steps, validationDataset_steps) = rawdata.randomSplit(Array(0.8, 0.2), 1234)
      var df_init = DataLoader.convertDFtoVecAssembly(trainingDataset_init, "features", "y")
      var df_step = DataLoader.convertDFtoVecAssembly(rawdata, "features", "y")


    BayeisanInitialStep(df_step, classifier, classifiermgr)
    for (i <- 1 to 10 )
      {
        BayeisanStep(spark , df_step , classifiermgr , i , classifier)
      }


    println("Best Accuracy found: " + ParamMapAccuracyLst.sortWith(_._2 > _._2)(0)._1)
    println("Best Parameters found: " + ParamMapAccuracyLst.sortWith(_._2 > _._2)(0)._2)

  }



  def learnRegression(df:DataFrame): RandomForestRegressionModel = {

    /*val lr = new LinearRegression()
      .setMaxIter(10)
      .setRegParam(0.3)
      .setElasticNetParam(0.8)*/

    val rf = new RandomForestRegressor()
      .setLabelCol("Label")
      .setFeaturesCol("features")

    val Array(training_Param, validation_Param) = df.randomSplit(Array(0.8, 0.2), 1234)
    val model = rf.fit(training_Param)

    var predictions = model.transform(validation_Param)

    val evaluator = new RegressionEvaluator()
      .setLabelCol("Label")
      .setPredictionCol("prediction")
      .setMetricName("rmse")
    val rmse = evaluator.evaluate(predictions)
    println("Regresssion Model root mean square error:" + rmse)
    return model
  }

  def getNextParameters(model:RandomForestRegressionModel , df_Param:DataFrame , df_data:DataFrame , step:Int, classifiermgr:ClassifiersManager , ClassifierName:String )
  {

    /*
    var maxBins = 0
    var minInstancesPerNode = 0
    var numTrees = 0
    var maxDepth = 0
    var minInfoGain = 0.0
    */

    var exist = false
    var accuracy = 0.0
    var acc = 0.0
    // Make predictions to get next param
    //val Array(training_Param, validation_Param) = df_Param.randomSplit(Array(0.001, 0.999), 1234)

    val predictions = model.transform(df_Param)
    var predicted_Param_accu = predictions.orderBy(desc("prediction")).collect().toList
    var currentVector: org.apache.spark.ml.linalg.DenseVector = null

    //var nextParamMap = predicted_Param_accu(0)(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector].toArray
    //var acc = predicted_Param_accu(0)(1).asInstanceOf[Double]

    //"maxBins", "minInstancesPerNode", "impurity", "numTrees", "maxDepth", "minInfoGain"
    //Check if the selected param already used

    breakable {
      for (i <- 0 to predicted_Param_accu.size - 1) {
        exist = false
        var currentParam = predicted_Param_accu(i)(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector]
        //nextParamMap = predicted_Param_accu(i)(0).asInstanceOf[org.apache.spark.ml.linalg.DenseVector].toArray
        //acc = predicted_Param_accu(0)(1).asInstanceOf[Double]

        ParamMapAccuracyLst.foreach(e => {
          exist=  compareRowtoParamMap(currentParam ,e._1 , ClassifierName )
        })
        if(! exist) {
          //println("Not Exist")
          //nextParamMap = currentParam.toArray
          currentVector = currentParam
          acc = predicted_Param_accu(i)(1).asInstanceOf[Double]
          break
        }
      }
    }
    //logger.close()

    println( "Selected Paramters: " + currentVector.toArray.toList + " expected accuracy:" + acc)
    var pm = convertVectortoParamMap(currentVector ,ClassifierName)

    // learn for the selected paramter and get actual accuracy
    /*
    val rf = new RandomForestClassifier()
      .setLabelCol("y")
      .setFeaturesCol("features")
    var pm = new ParamMap()
    pm.put(rf.maxBins,             nextParamMap(0).toInt)
    pm.put(rf.minInstancesPerNode, nextParamMap(1).toInt)
    pm.put(rf.numTrees,            nextParamMap(2).toInt)
    pm.put(rf.maxDepth,            nextParamMap(3).toInt)
    pm.put(rf.minInfoGain,         nextParamMap(4).toDouble)
    pm.put(rf.impurity,            "gini")

    var paramLst = new ListBuffer[ParamMap]()
    paramLst.append(pm)
    */
    //accuracy = singleLearn(df_data, pm, step, "RandomForestClassifier" , classifiermgr)


      accuracy = singleLearn(df_data, pm, step, ClassifierName, classifiermgr)
    try {
      /*
      var percent = 0.05 * (step+1)
      if(percent > 0.8)
        percent = 0.8

      val Array(trainingDataset, validationDataset) = df_data.randomSplit(Array(percent, 1- percent), 1234)
      singleLearn(df_data, pm , 1 , "RandomForestClassifier" , classifiermgr)
      //val Array(trainingDataset_1, validationDataset_1) = df_data.randomSplit(Array(0.3, 0.7), 1234)
      var rf_model = rf.fit(trainingDataset, pm).asInstanceOf[Model[_]]
      val rf_predictions = rf_model.transform(validationDataset)
      val evaluator = classifiermgr.evaluator
      accuracy = evaluator.evaluate(rf_predictions)
      println(" accuracy: --||-->>" + (accuracy * 100).toString)
      */
    }
    catch
      {
        case ex:Exception =>
          print("Exception")
      }
    //var ParamMapLst = new ListBuffer[ParamMap]()
    //ParamMapLst.append(pm)
    //val ParamMapAccuracyModelLst = learn(df_data, ParamMapLst.toList, "RandomForestClassifier" , classifiermgr)
   // ParamMapAccuracyModelLst.foreach( e => ParamMapAccuracyLst.append( (e._1 , e._2)) )

    ParamMapAccuracyLst.append( (pm ,accuracy * 100 ) )

  }

  def singleLearn(df_data: Dataset[_], paramMap:ParamMap , step:Int, ClassifierName:String , ClassifierMgr:ClassifiersManager ): Double =
  {

    var accuracy = 0.0
    var percent = 0.05 * (step+1)
    if(percent > 0.8)
      percent = 0.8
    val Array(trainingDataset, validationDataset) = df_data.randomSplit(Array(percent, 1- percent), 1234)


    var rf = createClassifier(paramMap , ClassifierName)
    var rf_model = rf.fit(trainingDataset).asInstanceOf[Model[_]]
    val rf_predictions = rf_model.transform(validationDataset)
    val evaluator = ClassifierMgr.evaluator
    accuracy = evaluator.evaluate(rf_predictions)
    println(" accuracy:-->>>>" + (accuracy * 100).toString)

    return accuracy

  }


  def createClassifier(paramMap:ParamMap , ClassifierName:String): Estimator[_] =
  {
    var estimator:Estimator[_] = null

    if( ClassifierName == "RandomForestClassifier") {
      var maxBins = 0
      var minInstancesPerNode = 0
      var numTrees = 0
      var maxDepth = 0
      var minInfoGain = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "numTrees")
          numTrees = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain = parmmap.value.asInstanceOf[Double]
      }

      val rf = new RandomForestClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMinInstancesPerNode(minInstancesPerNode)
        .setMinInfoGain(minInfoGain)
        .setMaxDepth(maxDepth)
        .setNumTrees(numTrees)
        .setMaxBins(maxBins)
      estimator = rf
      println("minInstancesPerNode:" + minInstancesPerNode + ", minInfoGain:" + minInfoGain + " ,:setMaxDepth" + maxDepth + " ,:numTrees" + numTrees + " ,:maxBins" + maxBins)
    }

    if( ClassifierName == "LogisticRegression") {
      var fitIntercept = false
      var maxIter = 0
      var regParam = 0.0
      var elasticNetParam = 0.0
      var standardization = false
      var tol = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "fitIntercept")
          fitIntercept = parmmap.value.asInstanceOf[Boolean]
        if (parmmap.param.name == "maxIter")
          maxIter = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "regParam")
          regParam = parmmap.value.asInstanceOf[Double]
        if (parmmap.param.name == "elasticNetParam")
          elasticNetParam = parmmap.value.asInstanceOf[Double]
        if (parmmap.param.name == "standardization")
          standardization = parmmap.value.asInstanceOf[Boolean]
        if (parmmap.param.name == "tol")
          tol = parmmap.value.asInstanceOf[Double]
      }
      val lr = new LogisticRegression()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setFitIntercept(fitIntercept)
        .setMaxIter(maxIter)
        .setRegParam(regParam)
        .setElasticNetParam(elasticNetParam)
        .setStandardization(standardization)
        .setTol(tol)
      estimator = lr
      println("fitIntercept:" + fitIntercept + ", maxIter:" + maxIter + " ,:regParam" + regParam +
              " ,:elasticNetParam" + elasticNetParam + " ,:standardization" + standardization + ",:tol" + tol)
    }

    if( ClassifierName == "DecisionTreeClassifier") {
      var maxBins = 0
      var minInstancesPerNode = 0
      var maxDepth = 0
      var minInfoGain = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain = parmmap.value.asInstanceOf[Double]
      }

      val dt = new DecisionTreeClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMinInstancesPerNode(minInstancesPerNode)
        .setMinInfoGain(minInfoGain)
        .setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
      estimator = dt
      println("minInstancesPerNode:" + minInstancesPerNode + ", minInfoGain:" + minInfoGain + " ,:setMaxDepth" + maxDepth + " ,:maxBins" + maxBins)

    }

    if ( ClassifierName == "GBTClassifier") {

      var maxBins = 0
      var minInstancesPerNode = 0
      var maxDepth = 0
      var minInfoGain = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain = parmmap.value.asInstanceOf[Double]
      }

      val gbt = new GBTClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMinInstancesPerNode(minInstancesPerNode)
        .setMinInfoGain(minInfoGain)
        .setMaxDepth(maxDepth)
        .setMaxBins(maxBins)
      estimator = gbt
      println("minInstancesPerNode:" + minInstancesPerNode + ", minInfoGain:" + minInfoGain + " ,:setMaxDepth" + maxDepth + " ,:maxBins" + maxBins)


    }

    if( ClassifierName == "LinearSVC") {
      var maxIter = 0
      var regParam = 0.0

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxIter")
          maxIter = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "regParam")
          regParam = parmmap.value.asInstanceOf[Double]
      }
      val svc= new LinearSVC()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMaxIter(maxIter)
        .setRegParam(regParam)
      estimator = svc
      println("maxIter:" + maxIter + " ,:regParam" + regParam )
    }

    if ( ClassifierName == "MultilayerPerceptronClassifier")
    {
      var maxIter = 0
      var Layers :Array[Int]= null

      for (i <- 0 to paramMap.toSeq.size - 1) {
        var parmmap = paramMap.toSeq(i)
        if (parmmap.param.name == "maxIter")
          maxIter = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "layers")
          Layers = parmmap.value.asInstanceOf[Array[Int]]
      }
      val mlp= new MultilayerPerceptronClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
        .setMaxIter(maxIter)
        .setLayers(Layers)
      estimator = mlp
      println("maxIter:" + maxIter + " ,:Layers" + Layers.toSeq )
    }
    return estimator
  }

  def convertVectortoParamMap(inputVector:org.apache.spark.ml.linalg.DenseVector , ClassifierName:String): ParamMap =
  {
    var pm = new ParamMap()
    var currentParam = inputVector.toArray
    if (ClassifierName == "RandomForestClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var numTrees = currentParam(2)
      var maxDepth = currentParam(3)
      var minInfoGain = currentParam(4)

      val rf = new RandomForestClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(rf.maxBins, maxBins.toInt)
      pm.put(rf.minInstancesPerNode, minInstancesPerNode.toInt)
      pm.put(rf.numTrees, numTrees.toInt)
      pm.put(rf.maxDepth, maxDepth.toInt)
      pm.put(rf.minInfoGain, minInfoGain.toDouble)
      pm.put(rf.impurity, "gini")
    }

    if (ClassifierName == "LogisticRegression"){
      var fitIntercept = false
      if(currentParam(0) == 1.0)
        fitIntercept = true
      var maxIter = currentParam(1)
      var regParam = currentParam(2)
      var elasticNetParam = currentParam(3)
      var standardization = false
      if(currentParam(4) == 1.0)
        standardization = true
      var tol = currentParam(5)

      val lr = new LogisticRegression()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(lr.fitIntercept , fitIntercept)
      pm.put(lr.maxIter , maxIter.toInt)
      pm.put(lr.regParam ,regParam )
      pm.put(lr.elasticNetParam ,elasticNetParam )
      pm.put(lr.standardization ,standardization )
      pm.put(lr.tol , tol)
    }

    if (ClassifierName == "DecisionTreeClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var maxDepth = currentParam(2)
      var minInfoGain = currentParam(3)

      val dt = new DecisionTreeClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(dt.maxBins, maxBins.toInt)
      pm.put(dt.minInstancesPerNode, minInstancesPerNode.toInt)
      pm.put(dt.maxDepth, maxDepth.toInt)
      pm.put(dt.minInfoGain, minInfoGain.toDouble)
      pm.put(dt.impurity, "gini")
    }

    if (ClassifierName == "GBTClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var maxDepth = currentParam(2)
      var minInfoGain = currentParam(3)

      val gbt = new GBTClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(gbt.maxBins, maxBins.toInt)
      pm.put(gbt.minInstancesPerNode, minInstancesPerNode.toInt)
      pm.put(gbt.maxDepth, maxDepth.toInt)
      pm.put(gbt.minInfoGain, minInfoGain.toDouble)
      pm.put(gbt.impurity, "gini")
    }

    if( ClassifierName == "LinearSVC"){
      var maxIter = currentParam(0)
      var regParam = currentParam(1)
      val svc = new LinearSVC()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(svc.maxIter , maxIter.toInt)
      pm.put(svc.regParam ,regParam )

    }

    if( ClassifierName == "MultilayerPerceptronClassifier"){
      var maxIter = currentParam(0)
      var Layers:Array[Int] =  Array(currentParam(1).toInt,currentParam(2).toInt,currentParam(3).toInt)
      val mlp = new MultilayerPerceptronClassifier()
        .setLabelCol("y")
        .setFeaturesCol("features")
      pm.put(mlp.maxIter , maxIter.toInt)
      pm.put(mlp.layers ,Layers )

    }

    return pm
  }
  def compareRowtoParamMap(inputVector:org.apache.spark.ml.linalg.DenseVector , pm:ParamMap, ClassifierName:String ): Boolean =
  {
    var currentParam = inputVector.toArray
    var result = false
    if (ClassifierName == "RandomForestClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var numTrees = currentParam(2)
      var maxDepth = currentParam(3)
      var minInfoGain = currentParam(4)

      var maxBins_b = 0
      var minInstancesPerNode_b = 0
      var numTrees_b = 0
      var maxDepth_b = 0
      var minInfoGain_b = 0.0

      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "numTrees")
          numTrees_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain_b = parmmap.value.asInstanceOf[Double]
      }

      if( maxBins == maxBins_b  &&
        minInstancesPerNode == minInstancesPerNode_b &&
        numTrees == numTrees_b &&
        maxDepth == maxDepth_b &&
        minInfoGain == minInfoGain_b)
        result = true
    }
    if (ClassifierName == "LogisticRegression") {
      var fitIntercept = false
      if(currentParam(0) == 1.0)
        fitIntercept = true
      var maxIter = currentParam(1)
      var regParam = currentParam(2)
      var elasticNetParam = currentParam(3)
      var standardization = false
      if(currentParam(4) == 1.0)
        standardization = true
      var tol = currentParam(5)

      var fitIntercept_b = false
      var maxIter_b = 0
      var regParam_b = 0.0
      var elasticNetParam_b = 0.0
      var standardization_b = false
      var tol_b = 0.0

      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "fitIntercept")
          fitIntercept_b = parmmap.value.asInstanceOf[Boolean]
        if (parmmap.param.name == "maxIter")
          maxIter_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "regParam")
          regParam_b = parmmap.value.asInstanceOf[Double]
        if (parmmap.param.name == "elasticNetParam")
          elasticNetParam_b = parmmap.value.asInstanceOf[Double]
        if (parmmap.param.name == "standardization")
          standardization_b = parmmap.value.asInstanceOf[Boolean]
        if (parmmap.param.name == "tol")
          tol_b = parmmap.value.asInstanceOf[Double]
      }

      if( fitIntercept == fitIntercept_b  &&
        maxIter == maxIter_b &&
        regParam == regParam_b &&
        elasticNetParam == elasticNetParam_b &&
        standardization == standardization_b &&
        tol == tol_b
      )
        result = true

    }
    if (ClassifierName == "DecisionTreeClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var maxDepth = currentParam(2)
      var minInfoGain = currentParam(3)

      var maxBins_b = 0
      var minInstancesPerNode_b = 0
      var maxDepth_b = 0
      var minInfoGain_b = 0.0

      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain_b = parmmap.value.asInstanceOf[Double]
      }

      if( maxBins == maxBins_b  &&
        minInstancesPerNode == minInstancesPerNode_b &&
        maxDepth == maxDepth_b &&
        minInfoGain == minInfoGain_b)
        result = true
    }
    if (ClassifierName == "GBTClassifier") {
      // Row
      var maxBins = currentParam(0)
      var minInstancesPerNode = currentParam(1)
      var maxDepth = currentParam(2)
      var minInfoGain = currentParam(3)

      var maxBins_b = 0
      var minInstancesPerNode_b = 0
      var maxDepth_b = 0
      var minInfoGain_b = 0.0

      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "maxBins")
          maxBins_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInstancesPerNode")
          minInstancesPerNode_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "maxDepth")
          maxDepth_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "minInfoGain")
          minInfoGain_b = parmmap.value.asInstanceOf[Double]
      }

      if( maxBins == maxBins_b  &&
        minInstancesPerNode == minInstancesPerNode_b &&
        maxDepth == maxDepth_b &&
        minInfoGain == minInfoGain_b)
        result = true
    }
    if (ClassifierName == "LinearSVC") {
      var maxIter = currentParam(0)
      var regParam = currentParam(1)
      var maxIter_b = 0
      var regParam_b = 0.0
      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "maxIter")
          maxIter_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "regParam")
          regParam_b = parmmap.value.asInstanceOf[Double]
      }

      if(
        maxIter == maxIter_b &&
        regParam == regParam_b
      )
        result = true

    }
    if (ClassifierName == "MultilayerPerceptronClassifier") {
      var maxIter = currentParam(0).toInt
      var Layer = currentParam(2).toInt
      var maxIter_b = 0
      var Layer_b = 0
      for (i <- 0 to pm.toSeq.size - 1) {
        var parmmap = pm.toSeq(i)
        if (parmmap.param.name == "maxIter")
          maxIter_b = parmmap.value.asInstanceOf[Int]
        if (parmmap.param.name == "layers") {
          Layer_b = parmmap.value.asInstanceOf[Array[Int]](1)
        }
      }

      if(
        maxIter == maxIter_b &&
          Layer == Layer_b
      )
        result = true

    }

    return result

  }

  def learn(dataset: Dataset[_], paramLst: List[ParamMap] , ClassifierName:String , ClassifierMgr:ClassifiersManager): ListBuffer[(ParamMap, Double, Model[_])] = {
    var result = new ListBuffer[(ParamMap, Double, Model[_])]()
    val Array(trainingDataset, validationDataset) = dataset.randomSplit(Array(0.2, 0.2), 1234)
    var rf = ClassifierMgr.ClassifiersMap(ClassifierName)


    println("   Start Learn ...")

      for (param <- paramLst) {
        var model = rf.fit(trainingDataset, param).asInstanceOf[Model[_]]
        val predictions = model.transform(validationDataset)
        val evaluator = ClassifierMgr.evaluator
        val accuracy = evaluator.evaluate(predictions)
        if(ClassifierName == "RandomForestClassifier")
          result.append((param, accuracy * 100, model.asInstanceOf[RandomForestClassificationModel]))
        if(ClassifierName == "LogisticRegression")
          result.append((param, accuracy * 100, model.asInstanceOf[LogisticRegressionModel]))
        if(ClassifierName == "DecisionTreeClassifier")
          result.append((param, accuracy * 100, model.asInstanceOf[DecisionTreeClassificationModel]))
        if( ClassifierName == "GBTClassifier")
          result.append((param, accuracy * 100, model.asInstanceOf[GBTClassificationModel]))
        if( ClassifierName == "LinearSVC")
          result.append((param, accuracy * 100, model.asInstanceOf[LinearSVCModel]))
        if( ClassifierName == "MultilayerPerceptronClassifier")
          result.append((param, accuracy * 100, model.asInstanceOf[MultilayerPerceptronClassificationModel]))

        println("Learned:")
        println(printParamMap(param))
        println(" -->  Acc:" + accuracy)
        //println("--------------------------------")
      }
    try {
    }
  catch
  {
    case ex:Exception =>
      print("Exception")
  }
    return result

  }

  def convertParamtoDF(spark:SparkSession,paramMapList:ListBuffer[(ParamMap, Double)], WithoutLabel:Boolean , Classifier:String): DataFrame =
  {
    import spark.implicits._
    if(Classifier == "RandomForestClassifier") {
      var RF_Param_Lst = new ListBuffer[(Int, Int, String, Int, Int, Double, Double)]()
      for (pm <- paramMapList) {
        var maxBins = 0
        var minInstancesPerNode = 0
        var impurity = ""
        var numTrees = 0
        var maxDepth = 0
        var minInfoGain = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxBins")
            maxBins = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInstancesPerNode")
            minInstancesPerNode = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "impurity")
            impurity = parmmap.value.asInstanceOf[String]
          if (parmmap.param.name == "numTrees")
            numTrees = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "maxDepth")
            maxDepth = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInfoGain")
            minInfoGain = parmmap.value.asInstanceOf[Double]
        }
        RF_Param_Lst.append((maxBins, minInstancesPerNode, impurity, numTrees, maxDepth, minInfoGain, pm._2))
      }
      var columns = Seq("maxBins", "minInstancesPerNode", "impurity", "numTrees", "maxDepth", "minInfoGain" , "Label")
      val data = RF_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("maxBins", "minInstancesPerNode", "impurity", "numTrees", "maxDepth", "minInfoGain" , "Label")
      val indexer = new StringIndexer().setInputCol("impurity").setOutputCol("c_impurity").fit(df)
      df = indexer.transform(df).drop("impurity")
      if(WithoutLabel)
        df.drop("Label")
      return df
    }
    if(Classifier == "LogisticRegression"){
      var LR_Param_Lst = new ListBuffer[(Boolean, Int, Double, Double, Boolean, Double, Double)]()
      //fitIntercept (Boolean) - maxIter (Int) - regParam (Double)- elasticNetParam (Double)- standardization (bool)- tol (Double)
      for (pm <- paramMapList) {
        var fitIntercept = false
        var maxIter = 0
        var regParam = 0.0
        var elasticNetParam = 0.0
        var standardization = false
        var tol = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "fitIntercept")
            fitIntercept = parmmap.value.asInstanceOf[Boolean]
          if (parmmap.param.name == "maxIter")
            maxIter = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "regParam")
            regParam = parmmap.value.asInstanceOf[Double]
          if (parmmap.param.name == "elasticNetParam")
            elasticNetParam = parmmap.value.asInstanceOf[Double]
          if (parmmap.param.name == "standardization")
            standardization = parmmap.value.asInstanceOf[Boolean]
          if (parmmap.param.name == "tol")
            tol = parmmap.value.asInstanceOf[Double]
        }
        LR_Param_Lst.append((fitIntercept,maxIter,regParam,elasticNetParam,standardization,tol,pm._2))
      }
      val data = LR_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("fitIntercept", "maxIter", "regParam", "elasticNetParam", "standardization", "tol" , "Label")

//      val indexer1 = new StringIndexer().setInputCol("fitIntercept").setOutputCol("c_fitIntercept").fit(df)
//      df = indexer1.transform(df).drop("fitIntercept")
//      val indexer2 = new StringIndexer().setInputCol("standardization").setOutputCol("c_standardization").fit(df)
//      df = indexer2.transform(df).drop("standardization")
      if(WithoutLabel)
        df.drop("Label")
      return df

    }
    if(Classifier == "DecisionTreeClassifier") {
      var RF_Param_Lst = new ListBuffer[(Int, Int, String, Int, Double, Double)]()
      for (pm <- paramMapList) {
        var maxBins = 0
        var minInstancesPerNode = 0
        var impurity = ""
        var maxDepth = 0
        var minInfoGain = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxBins")
            maxBins = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInstancesPerNode")
            minInstancesPerNode = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "impurity")
            impurity = parmmap.value.asInstanceOf[String]
          if (parmmap.param.name == "maxDepth")
            maxDepth = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInfoGain")
            minInfoGain = parmmap.value.asInstanceOf[Double]
        }
        RF_Param_Lst.append((maxBins, minInstancesPerNode, impurity, maxDepth, minInfoGain, pm._2))
      }
      var columns = Seq("maxBins", "minInstancesPerNode", "impurity",  "maxDepth", "minInfoGain" , "Label")
      val data = RF_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("maxBins", "minInstancesPerNode", "impurity",  "maxDepth", "minInfoGain" , "Label")
      val indexer = new StringIndexer().setInputCol("impurity").setOutputCol("c_impurity").fit(df)
      df = indexer.transform(df).drop("impurity")
      if(WithoutLabel)
        df.drop("Label")
      return df
    }
    if(Classifier == "GBTClassifier") {
      var RF_Param_Lst = new ListBuffer[(Int, Int, String, Int, Double, Double)]()
      for (pm <- paramMapList) {
        var maxBins = 0
        var minInstancesPerNode = 0
        var impurity = ""
        var maxDepth = 0
        var minInfoGain = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxBins")
            maxBins = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInstancesPerNode")
            minInstancesPerNode = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "impurity")
            impurity = parmmap.value.asInstanceOf[String]
          if (parmmap.param.name == "maxDepth")
            maxDepth = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "minInfoGain")
            minInfoGain = parmmap.value.asInstanceOf[Double]
        }
        RF_Param_Lst.append((maxBins, minInstancesPerNode, impurity, maxDepth, minInfoGain, pm._2))
      }
      var columns = Seq("maxBins", "minInstancesPerNode", "impurity",  "maxDepth", "minInfoGain" , "Label")
      val data = RF_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("maxBins", "minInstancesPerNode", "impurity",  "maxDepth", "minInfoGain" , "Label")
      val indexer = new StringIndexer().setInputCol("impurity").setOutputCol("c_impurity").fit(df)
      df = indexer.transform(df).drop("impurity")
      if(WithoutLabel)
        df.drop("Label")
      return df
    }
    if(Classifier == "LinearSVC"){
      var LR_Param_Lst = new ListBuffer[(Int, Double, Double)]()
      //fitIntercept (Boolean) - maxIter (Int) - regParam (Double)- elasticNetParam (Double)- standardization (bool)- tol (Double)
      for (pm <- paramMapList) {
        var maxIter = 0
        var regParam = 0.0
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxIter")
            maxIter = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "regParam")
            regParam = parmmap.value.asInstanceOf[Double]
        }
        LR_Param_Lst.append((maxIter,regParam,pm._2))
      }
      val data = LR_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF("maxIter", "regParam",  "Label")

      //      val indexer1 = new StringIndexer().setInputCol("fitIntercept").setOutputCol("c_fitIntercept").fit(df)
      //      df = indexer1.transform(df).drop("fitIntercept")
      //      val indexer2 = new StringIndexer().setInputCol("standardization").setOutputCol("c_standardization").fit(df)
      //      df = indexer2.transform(df).drop("standardization")
      if(WithoutLabel)
        df.drop("Label")
      return df

    }
    if(Classifier == "MultilayerPerceptronClassifier"){
      var P_Param_Lst = new ListBuffer[(Int, Int,Int,Int, Double)]()
      var layers_0 = 0
      var layers_1 = 0
      var layers_2 = 0
      var maxIter = 0
      for (pm <- paramMapList) {
        for (i <- 0 to pm._1.toSeq.size - 1) {
          var parmmap = pm._1.toSeq(i)
          if (parmmap.param.name == "maxIter")
            maxIter = parmmap.value.asInstanceOf[Int]
          if (parmmap.param.name == "layers") {
            layers_0 = parmmap.value.asInstanceOf[Array[Int]](0)
            layers_1 = parmmap.value.asInstanceOf[Array[Int]](1)
            layers_2 = parmmap.value.asInstanceOf[Array[Int]](2)
          }
        }
        P_Param_Lst.append((maxIter,layers_0,layers_1,layers_2,pm._2))
      }
      val data = P_Param_Lst.toSeq
      val rdd = spark.sparkContext.parallelize(data)
      var df = rdd.toDF( "maxIter","layers_0" ,"layers_1" ,"layers_2" ,  "Label")
      if(WithoutLabel)
        df.drop("Label")
      return df

      }
    return null

  }

  def BayeisanInitialStep(df:DataFrame , classifierName:String,  classifiermgr:ClassifiersManager): Unit =
  {
    println("Start Initial Step...")
    println("---------------------")
    val ParamMapAccuracyModelLst = learn(df, paramMapList.toList, classifierName , classifiermgr)
    ParamMapAccuracyModelLst.foreach( e => ParamMapAccuracyLst.append( (e._1 , e._2)) )

    logger.logOutput("1 - Initial Random Paramerter and their accuracy...\n")
    ParamMapAccuracyModelLst.foreach( e => {
      logger.logOutput(printParamMap(e._1))
      logger.logOutput( "Accuracy: " + e._2.toString + "\n") })
    logger.logOutput("---------------------------------------------------------------------------------")
    println("-----------------------------------------------------------------------------")

  }

  def BayeisanStep(spark:SparkSession , df_data:DataFrame, classifiermgr:ClassifiersManager , step:Int, classifier:String = "RandomForestClassifier"): Unit =
  {
    println("start Bayesian Step:" + (step) )
    println("---------------------")
    //convert result to DF
    var rdf = convertParamtoDF(spark,ParamMapAccuracyLst , false,classifier)

    // convert Result to Assembler
    var bdf = DataLoader.convertDFtoVecAssembly( rdf , "features" , "Label")


    //Call Regession
    val m = learnRegression(bdf)

    getNextParameters(m, Allparam_df, df_data , step, classifiermgr,classifier)

    //println(ParamMapAccuracyLst)
    println("-----------------------------------------------------------------------------")
  }

  def printParamMap( pm:ParamMap): String =
  {
    var result = ""
    for ( p <- pm.toSeq)
      {
        result = result +  p.param.name + ":" + p.value + " - "
      }
    //print(result)
    return result
  }



}
