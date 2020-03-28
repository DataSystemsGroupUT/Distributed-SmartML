package org.apache.spark.ml.tuning

import java.util.Date

/**
  * Helper Class for Optimizer
  */
object TuningHelper {

  /**
    * check if timeout or not
    * @return
    */
  def IsTimeOut(maxTime:Long ,StartingTime:Date ): Boolean = {
    if (getRemainingTime(maxTime,StartingTime ) == 0)
      return true
    else
      return false
  }

  /**
    * get remaining time
    * @return
    */
  def getRemainingTime(maxTime:Long ,StartingTime:Date ): Long = {
    var rem: Long = ( maxTime * 1000) - (new Date().getTime - StartingTime.getTime())
    if (rem < 0)
      rem = 0
    return rem
  }

  /**
    * get elapsed time
    * @return
    */
  def getElapsedTime(localStartTime:Date):Long ={
    return (new Date().getTime - localStartTime.getTime())
  }


}
