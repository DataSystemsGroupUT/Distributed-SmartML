package org.apache.spark.ml.classification

import org.apache.hadoop.fs.Path
import org.apache.spark.SparkContext
import org.apache.spark.annotation.Since
import org.apache.spark.ml.PredictorParams
import org.apache.spark.ml.feature.StandardScalerModel
import org.apache.spark.ml.linalg.{BLAS, DenseVector, Vector, Vectors}
import org.apache.spark.ml.param.{BooleanParam, ParamMap}
import org.apache.spark.ml.util._
import org.apache.spark.mllib.linalg.distributed.RowMatrix
import org.apache.spark.sql.functions.{col, lit}
import org.apache.spark.sql.{Dataset}


/**
  **********************************************************************************************************************
  * Params for QDA Classifiers.
  **********************************************************************************************************************
*/
@Since("2.1.0")
private[classification] trait QDAParams extends PredictorParams  {


  /**
    * The Scaled Data parameter.
    * (default = 1.0).
    * @group param
    */
  final val scaledData: BooleanParam = new BooleanParam(this, "scaledData", "is the input data scaled ? or we should scale it")

  /** @group getParam */
  final def getscaled_data: Boolean = $(scaledData)
}

/**
 ***********************************************************************************************************************
 QDA Algorithm Implementation (Estimator)
 ***********************************************************************************************************************
 */
@Since("2.1.0")
class QDA (override val uid: String , sc: SparkContext)
  extends ProbabilisticClassifier[Vector, QDA, QDAModel]
    with QDAParams with DefaultParamsWritable {

  def this( sc: SparkContext) = this(Identifiable.randomUID("qda") , sc )
  def setScaledData(value: Boolean): this.type = set(scaledData, value)
  setDefault(scaledData -> false)

  /**
    * train function
  */
  @Since("2.1.0")
  override protected def train(dataset: Dataset[_]): QDAModel = {

    var InvMainCovarianceMatrix:org.apache.spark.mllib.linalg.DenseMatrix = null
    var meanMap: Map[Int ,Vector] = null
    var ClassesProbMap : Map[Int , Double] = null
    var CovDetMap : Map[Int , Double] = null
    var CovInvMatrixMap : Map[Int , org.apache.spark.mllib.linalg.DenseMatrix] = null
    var totalObservations:Double = 0.0
    var scalerModel : StandardScalerModel = null
    var util: LDAUtil = new LDAUtil()

    // Get the number of classes.
    val numClasses = getNumClasses(dataset)

    // log
    val instr = Instrumentation.create(this, dataset)
    instr.logParams(labelCol, featuresCol,  predictionCol, rawPredictionCol,
      probabilityCol, scaledData, thresholds)

    //get number of features
    val numFeatures = dataset.select(col($(featuresCol))).head().getAs[Vector](0).size
    instr.logNumFeatures(numFeatures)


    //calculate statistics from the dataset (count and sum per class)
    val aggregated = dataset.select(col($(labelCol)), lit(1.0), col($(featuresCol))).rdd
      .map { row => (row.getDouble(0), (row.getDouble(1), row.getAs[Vector](2)))
      }.aggregateByKey[(Double, DenseVector)]((0.0, Vectors.zeros(numFeatures).toDense))(
      seqOp = {
        case ((countSum: Double, featureSum: DenseVector), (count, features)) =>
          //requireValues(features)
          BLAS.axpy(1.0, features , featureSum)
          //BLAS.scal( 1.0, featureSum)
          (countSum + count, featureSum)
      },
      combOp = {
        case ((countSum1, featureSum1), (countSum2, featureSum2)) =>
          BLAS.axpy(1.5, featureSum2, featureSum1)
          //BLAS.scal( 0.5 , featureSum1)
          (countSum1 + countSum2,  featureSum1)
      }).collect().sortBy(_._1)

    // get number of classes
    val numLabels = aggregated.length

        // calculate covariance matrix (inverse and Determinant) per class (using Distributed Row Matrix)
    var i = 0
    aggregated.foreach { case (label, (n, sumTermFreqs)) =>
      println("Lable:" +  label)

      // filter the dataset by class and get a vector of features
      val ds = dataset.select(col($(labelCol)), col($(featuresCol)))
        .filter($(labelCol)+ "== " + label).rdd
        .map( row =>  row.getAs[org.apache.spark.ml.linalg.Vector](1).toDense )

      // convert the filtered dataset into a distributed matrix
      val rowMatrix = new RowMatrix( ds.map(r=> org.apache.spark.mllib.linalg.DenseVector.fromML(r)))

      // calculate the covariance matrix
      var CovarianceMatrix : org.apache.spark.mllib.linalg.Matrix = rowMatrix.computeCovariance()

      // inverse the co-variance matrix and get its determinaint
      var ( covarianceMatrixInverse : org.apache.spark.mllib.linalg.DenseMatrix , covarianceMatrixDet : Double) =
                        util.denseMatrixInverseAndDet(CovarianceMatrix, sc)

      if( i ==0 )
       {
         CovInvMatrixMap = Map( label.toInt ->   covarianceMatrixInverse)
         CovDetMap = Map( label.toInt ->  covarianceMatrixDet)
       }
      else
      {
        CovInvMatrixMap += ( label.toInt ->  covarianceMatrixInverse )
        CovDetMap += ( label.toInt ->  covarianceMatrixDet)
      }

      // accumulate to get the total number of observations
      totalObservations = totalObservations + n
      i += 1
    }
    // create map containg the mean vector for each class
    meanMap = aggregated.map( r => (r._1.toInt -> new DenseVector(r._2._2.toArray.map( e => e /r._2._1 )))).toMap

    //create map contains the probability (Nk/N) per each class
    ClassesProbMap = aggregated.map( r => (r._1.toInt -> r._2._1.toDouble / totalObservations)).toMap

    //create an QDA Model instance
    new QDAModel(uid, CovInvMatrixMap , CovDetMap , meanMap , ClassesProbMap , getscaled_data )}

  override def copy(extra: ParamMap): QDA = defaultCopy(extra)
}

object QDA extends DefaultParamsReadable[QDA] {

  override def load(path: String): QDA = super.load(path)
}


/**
 ***********************************************************************************************************************
 QDA Model Implementation (Transformer)
 ***********************************************************************************************************************
 */
@Since("2.1.0")
class QDAModel private[ml] (override val uid: String,
                            val invCovMatrixMap: Map[Int , org.apache.spark.mllib.linalg.DenseMatrix],
                            val covDetMap:  Map[Int , Double] ,
                            val classesMean: Map[Int, Vector] ,
                            val classesProb:Map[Int,Double] ,
                            val dataScaled: Boolean)

  extends ProbabilisticClassificationModel[Vector, QDAModel]
    with QDAParams with MLWritable {


  override val numFeatures: Int = classesMean(1).size// invCovMatrixMap.numCols
  override val numClasses: Int = classesProb.keys.size
  override def write: MLWriter = new QDAModel.QDAModelWriter(this)

  /**
    * predict function
  */
  @Since("2.1.0")
  override protected def predictRaw(features: Vector): Vector = {
    var Delta: Double = 0
    var count: Int = 0
    var rawPrediction: Vector = null
    var result: Array[Double] = Array.fill(numClasses)(0)

    // Assume we have n features so ,
    //--------------------------------------------------------
    // 1- point X: is a vector of n element = Matrix ( n x 1 )
    // 2- point x transposed : is Matrix ( 1 x n )
    // 3- mean : is a vector of n element = Matrix ( n x 1 )
    // 4- mean transpose : is Matrix ( 1 x n )
    // 5- co-variance matrix inverse: is Matrix ( n x n )


    //convert the point x to matrix  and matrix transposed [x1, x2, x3, ...xn]
    var tran_x = new org.apache.spark.mllib.linalg.DenseMatrix(1, numFeatures, features.toArray) //f.drop(1).dropRight(1).split(',').map(_.toDouble))
    var x = new org.apache.spark.mllib.linalg.DenseMatrix(numFeatures,1, features.toArray) //f.drop(1).dropRight(1).split(',').map(_.toDouble))

    // loop for each class and calculate the delta, chose class with highest delta
    for (k <- 0 to numClasses - 1)
    {
      // convert mean vector to DenseMatrix
      var meanmatrix = new org.apache.spark.mllib.linalg.DenseMatrix(numFeatures, 1, classesMean(k).toArray)

      //Get the mean matrix transepose
      var tran_meanmatrix = new org.apache.spark.mllib.linalg.DenseMatrix(1, numFeatures, classesMean(k).toArray)

      // Inversed Covariance matrix
      val Inv_Cov : org.apache.spark.mllib.linalg.DenseMatrix = invCovMatrixMap(k)

      //calculate Delta
      val partOne:   Double = -0.5 * ((tran_x.multiply(Inv_Cov)).multiply(x)) (0, 0)
      val partTwo:   Double = ((tran_x.multiply(Inv_Cov)).multiply(meanmatrix)) (0, 0)
      val partThree: Double = -0.5 * (((tran_meanmatrix.multiply(Inv_Cov)).multiply(meanmatrix)) (0, 0))
      var partFour:  Double = -0.5 * math.log10(covDetMap(k))
      var partFive:  Double = math.log10(classesProb(k))
      var Currdelta: Double = partOne + partTwo + partThree + partFour + partFive
      result(count) = Currdelta
      count += 1
    }
    rawPrediction = new DenseVector(result)
    rawPrediction
  }

  /**
    * predict with probability
  */
  @Since("2.1.0")
  override protected def raw2probabilityInPlace(rawPrediction: Vector): Vector ={
    var total = 0.0
    var negcomp = 0.0
    var deltas = rawPrediction.toArray

    negcomp = deltas.map( e => {
      if (e < 0 ) (- 2 * e)
      else 0
    }).sum

    deltas = deltas.map( e=> e + negcomp)

    total = deltas.sum
    // deltas = deltas.map(e => (e.toDouble /total.toDouble) * 100)
    deltas = deltas.map( e =>  (  (  e.toDouble /total.toDouble) * 100) )

    return new DenseVector(deltas)
  }

  // copy
  override def copy(extra: ParamMap): QDAModel = {
    null
    //copyValues(new QDAModel(uid, pi, theta).setParent(this.parent), extra)
  }

}


object QDAModel extends MLReadable[QDAModel] {

  override def read: MLReader[QDAModel] = new QDAModelReader

  override def load(path: String): QDAModel = super.load(path)

  /** [[MLWriter]] instance for [[QDAModel]] */
  private[QDAModel] class QDAModelWriter(instance: QDAModel) extends MLWriter {

    private case class Data( invCovMatrixMap: Map[Int , org.apache.spark.mllib.linalg.DenseMatrix],
                             covDetMap:  Map[Int , Double] ,
                             classesMean: Map[Int, Vector] ,
                             classesProb:Map[Int,Double] ,
                             dataScaled: Boolean)

    override protected def saveImpl(path: String): Unit = {

      // Save metadata and Params
      DefaultParamsWriter.saveMetadata(instance, path, sc)
      // Save model data: pi, theta
      val data = Data(instance.invCovMatrixMap, instance.covDetMap , instance.classesMean, instance.classesProb , instance.dataScaled)
      val dataPath = new Path(path, "data").toString
      sparkSession.createDataFrame(Seq(data)).repartition(1).write.parquet(dataPath)

    }
  }

  @Since("2.1.0")
  private class QDAModelReader extends MLReader[QDAModel] {

    /** Checked against metadata when loading model */
    private val className = classOf[QDAModel].getName

    override def load(path: String): QDAModel = {


      val metadata = DefaultParamsReader.loadMetadata(path, sc, className)

      val dataPath = new Path(path, "data").toString

      val data = sparkSession.read.parquet(dataPath)
        .select("invCovMatrixMap", "covDetMap" , "classesMean" , "classesProb" , "dataScaled" ).head()
      val invCovMatrixMap = data.getAs[Map[Int , org.apache.spark.mllib.linalg.DenseMatrix] ](0)
      val covDetMap = data.getAs[Map[Int , Double]](1)
      val classesMean = data.getAs[Map[Int, Vector]](2)
      val classesProb = data.getAs[Map[Int,Double]](3)
      val dataScaled = data.getAs[Boolean](4)


      val model = new QDAModel(metadata.uid, invCovMatrixMap,covDetMap,classesMean,classesProb,dataScaled)

      DefaultParamsReader.getAndSetParams(model, metadata)
      model

    }
  }


}
