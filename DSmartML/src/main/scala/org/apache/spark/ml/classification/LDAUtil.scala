package org.apache.spark.ml.classification

import org.apache.spark.SparkContext
import org.apache.spark.mllib.linalg.distributed.RowMatrix

/**
  * Utitly Class to do Matrix operations (inverse, distribute, calculate Det,..)
  * RowMatrix methods return MLlib Matrix, so the utitlit use org.apache.spark.mllib.linalg.Matrix
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/3/2019
 */
class LDAUtil
{
  /**
    *  this function  Inverse DenseMatrix and get its determininat using SVD
    * @param m matrix to invers
    * @param sc spark Context object
    * @return the inversed Dense Matrix
    */
  def denseMatrixInverseAndDet(m: org.apache.spark.mllib.linalg.Matrix , sc: SparkContext) : (org.apache.spark.mllib.linalg.DenseMatrix, Double) = {
    computeInverseAndDeterminant(denseMatrixToRowMatrix(m,sc))
  }

  /**
    * this function Inverse DenseMatrix using SVD
    * @param m matrix to invers
    * @param sc spark Context object
    * @return he inversed Dense Matrix
    */
  def denseMatrixInverse(m: org.apache.spark.mllib.linalg.Matrix , sc: SparkContext) : org.apache.spark.mllib.linalg.Matrix = {
    computeInverse(denseMatrixToRowMatrix(m,sc))
  }


  /**
    * this function convert DenseMatrix to RowMatrix
    * @param m matrix to convert
    * @param sc spark Context object
    * @return converted matrix
    */
  def denseMatrixToRowMatrix(m: org.apache.spark.mllib.linalg.Matrix , sc: SparkContext) : RowMatrix = {
    new org.apache.spark.mllib.linalg.distributed.RowMatrix(matrixToRDD(m,sc))
  }

  /**
    * this function convert Matrix to RDD
     * @param m atrix to convert
    * @param sc spark Context object
    * @return converted matrix
    */
  def matrixToRDD(m: org.apache.spark.mllib.linalg.Matrix , sc: SparkContext) = {
    val columns = m.toArray.grouped(m.numRows)
    val rows = columns.toSeq.transpose // Skip this if you want a column-major RDD.
    val vectors = rows.map(row =>  org.apache.spark.mllib.linalg.Vectors.dense(row.toArray))
    LDAUtil.sc.parallelize(vectors)
  }

  /**
    * this function Calculate Matrix Inverse
    * @param X Row Matrix
    * @return Inversed Matrix
    */
  def computeInverse(X: RowMatrix): org.apache.spark.mllib.linalg.DenseMatrix = {
    val nCoef = X.numCols.toInt
    val svd = X.computeSVD(nCoef, computeU = true , 0.00000000001)  //Double.MinPositiveValue
    if (svd.s.size < nCoef) {
      sys.error(s"RowMatrix.computeInverse called on singular matrix.")

    }

    // Create the inv diagonal matrix from S
    val invS = org.apache.spark.mllib.linalg.DenseMatrix.diag(new org.apache.spark.mllib.linalg.DenseVector(svd.s.toArray.map(x => math.pow(x,-1))))

    // U cannot be a RowMatrix
    val U = new org.apache.spark.mllib.linalg.DenseMatrix(svd.U.numRows().toInt,svd.U.numCols().toInt,svd.U.rows.collect.flatMap(x => x.toArray))

    // If you could make V distributed, then this may be better. However its alreadly local...so maybe this is fine.
    val V = svd.V
    // inv(X) = V*inv(S)*transpose(U)  --- the U is already transposed.
    (V.multiply(invS)).multiply(U)
  }


  /**
    * this function Calculate Matrix Inverse
    * @param X Row Matr
    * @return Inversed Matrix
    */
  def computeInverseAndDeterminant(X: RowMatrix): (org.apache.spark.mllib.linalg.DenseMatrix, Double) = {
    val nCoef = X.numCols.toInt
    val svd = X.computeSVD(nCoef, computeU = true , 1E-18)  //Double.MinPositiveValue
    if (svd.s.size < nCoef) {
      sys.error(s"RowMatrix.computeInverse called on singular matrix.")

    }
    var Det : Double = svd.s.toArray.filter( x => x > 0 ).fold(1.0)(_ * _)
    // Create the inv diagonal matrix from S
    val invS = org.apache.spark.mllib.linalg.DenseMatrix.diag(new org.apache.spark.mllib.linalg.DenseVector(svd.s.toArray.map(x => math.pow(x,-1))))

    // U cannot be a RowMatrix
    val U = new org.apache.spark.mllib.linalg.DenseMatrix(svd.U.numRows().toInt,svd.U.numCols().toInt,svd.U.rows.collect.flatMap(x => x.toArray))

    // If you could make V distributed, then this may be better. However its alreadly local...so maybe this is fine.
    val V = svd.V
    // inv(X) = V*inv(S)*transpose(U)  --- the U is already transposed.
    ( (V.multiply(invS)).multiply(U) , Det)
  }


  /**
    * this function Calculate Matrix Determiniant
    * @param X RowMatrix
    * @return the matrix Determiniant
    */
   def computeDeterminant(X: RowMatrix):  Double = {
    val nCoef = X.numCols.toInt
    val svd = X.computeSVD(nCoef, computeU = true , 1E-9)  //Double.MinPositiveValue

    var Det : Double = svd.s.toArray.filter( x => x > 0 ).fold(1.0)(_ * _)
    //println("Determinant: " + Det)
    Det
  }


}

object LDAUtil
{
  var sc: SparkContext = null
}
