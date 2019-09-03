//package org.apache.spark.ml.tuning

package org.dsmartml

import java.io.{File, FileOutputStream, PrintWriter}


/**
  * Class Name: Logger
  * Description: this calss responable for logging information to text file
  *            :  it log (Time Log, Exception and Knowledgebase output)
  * @constructor Create a Logger object that allow us to log inforamtion to text file
  * @param path the path of the log files
  * @author Ahmed Eissa
  * @version 1.0
  * @Date 22/3/2019
  */
class Logger(path:String) extends java.io.Serializable {

  val total_width = 100
  /**
    * the name of the time log file
    */
  var timeLogFile = "timelog.txt"
  /**
    * the name of the Exception Log file
    */
  var exceptionLogFile = "Exlog.txt"
  /**
    * the name of the Knowledgebase file
    */
  var KBfile1 = "KB1_.csv"
  /**
    * the name of the Knowledgebase file
    */
  var KBfile2 = "KB2_.csv"


  var outfile = "output.txt"
  // Time Log File

  val file1: File = new File(path+timeLogFile)
  if(!file1.exists())
    File.createTempFile(path+timeLogFile , "txt")
  val pw1 = new PrintWriter(new FileOutputStream(file1 , true))

  // Excption Log File
  val file2: File = new File(path+exceptionLogFile)
  if(!file2.exists())
    File.createTempFile(path+exceptionLogFile , "txt")
  val pw2 = new PrintWriter(new FileOutputStream(file2 , true))

  //Knowledgebase File Name
  val file3: File = new File(path+KBfile1)
  if(!file3.exists())
    File.createTempFile(path+KBfile1 , "txt")
  val pw3 = new PrintWriter(new FileOutputStream(file3 , true))

  //Knowledgebase File Name
  val file4: File = new File(path+KBfile2)
  if(!file4.exists())
    File.createTempFile(path+KBfile2 , "txt")
  val pw4 = new PrintWriter(new FileOutputStream(file4 , true))

  //output
  val file5: File = new File(path+outfile)
  if(!file5.exists())
    File.createTempFile(path+outfile , "txt")
  val pw5 = new PrintWriter(new FileOutputStream(file5 , true))


  /**
    * Log information in the time log file
    * @param s information to be logged
    */
  def logTime(s:String)= {
    pw1.append(s)
  }

  /**
    * Log information in the Exception log file
    * @param s information to be logged
    */
  def logException(s:String)= {
    pw2.append(s)

  }

  /**
    * Log information in the KB file
    * @param s information to be logged
    */
  def logResult(s:String)= {
    pw3.append(s)

  }

  /**
    * Log information in the KB file
    * @param s information to be logged
    */
  def logResult1(s:String)= {
    pw4.append(s)
  }

  /**
    * Log information in the KB file
    * @param s information to be logged
    */
  def logOutput(s:String)= {
    pw5.append(s)
  }

  /**
    * Close File Writer
    */
  def close()={
    pw1.close()
    pw2.close()
    pw3.close()
    pw4.close()
    pw5.close()
  }


  def header_lr_space( len :Int): (Int,Int) =
  {

    var l1_s1 = 0
    var l1_s2 = 0
    if ( (total_width - len) % 2 == 0)
    {
      l1_s1 = (total_width - len)/2

      l1_s2 = (total_width - len)/2
    }
    else
    {
      l1_s1 = (total_width - len -1)/2
      l1_s2 = (total_width - len -1)/2 + 1
    }
    return (l1_s1, l1_s2)
  }

  def printHeader(HPOptimizer:Int, HP_MaxTime:Long,Parallelism:Int,eta:Int, maxResourcePercentage:Int,MaxRandomSearchParam:Int ): Unit =
  {
    println("-" * (total_width + 2))
    var title1 = "Smart ML 1.0"
    println("|"  + " " * header_lr_space(title1.length)._1 +  title1 + " " * header_lr_space(title1.length)._2 + "|")
    var title2 = "============"
    println("|"  + " " * header_lr_space(title2.length)._1 +  title2 + " " * header_lr_space(title2.length)._2 + "|")
    print("|")
    print(" " * (total_width))
    println("|")

    var dft:java.time.format.DateTimeFormatter = java.time.format.DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
    var dt = dft.format(java.time.LocalDateTime.now())
    var l1 = "Starting Date Time :" + dt
    var l2 = "Time Budget:" + HP_MaxTime + " second ,Models Parallelism:" + Parallelism
    var l3 = ""
    if (HPOptimizer==1) {
      l3 = "Hyper-parameter Optimizer: Hyperband, Eta:" + eta + ", Max Data Percentage Used:" + maxResourcePercentage + "%"
    }else
    {
      l3 = "Hyper-parameter Optimizer: Random Search, Max num of Models Trained:" + MaxRandomSearchParam
    }

    println("|"  + " " * header_lr_space(l1.length)._1 +  l1 + " " * header_lr_space(l1.length)._2 + "|")
    println("|"  + " " * header_lr_space(l1.length)._1 +  ("-" * l1.length) + " " * header_lr_space(l1.length)._2 + "|")

    println("|-" + l2 + " " * (((total_width ) - l2.length) - 1)  + "|")
    println("|-" + l3 + " " * (((total_width ) - l3.length) - 1)  + "|")
    println("-" * (total_width + 2))
  }

  def printLine(): Unit =
  {
    println("-" * (total_width + 2))
  }

  def logLine()={
    logOutput("-" * (total_width + 2) + "\n")
  }

  def logHeader(HPOptimizer:Int, HP_MaxTime:Long,Parallelism:Int,eta:Int, maxResourcePercentage:Int,MaxRandomSearchParam:Int )= {
    logOutput("-" * (total_width + 2) + "\n")
    var title1 = "Smart ML 1.0"
    logOutput("|"  + " " * header_lr_space(title1.length)._1 +  title1 + " " * header_lr_space(title1.length)._2 + "|\n")
    var title2 = "============"
    logOutput("|"  + " " * header_lr_space(title2.length)._1 +  title2 + " " * header_lr_space(title2.length)._2 + "|\n")
    logOutput("|")
    logOutput(" " * (total_width))
    logOutput("|\n")

    var dft:java.time.format.DateTimeFormatter = java.time.format.DateTimeFormatter.ofPattern("yyyy/MM/dd HH:mm:ss");
    var dt = dft.format(java.time.LocalDateTime.now())
    var l1 = "Starting Date Time :" + dt
    var l2 = "Time Budget:" + HP_MaxTime + " second ,Models Parallelism:" + Parallelism
    var l3 = ""
    if (HPOptimizer==1) {
      l3 = "Hyper-parameter Optimizer: Hyperband, Eta:" + eta + ", Max Data Percentage Used:" + maxResourcePercentage + "%"
    }else
    {
      l3 = "Hyper-parameter Optimizer: Random Search, Max num of Models Trained:" + MaxRandomSearchParam
    }

    logOutput("|"  + " " * header_lr_space(l1.length)._1 +  l1 + " " * header_lr_space(l1.length)._2 + "|\n")
    logOutput("|"  + " " * header_lr_space(l1.length)._1 +  ("-" * l1.length) + " " * header_lr_space(l1.length)._2 + "|\n")

    logOutput("|-" + l2 + " " * (((total_width ) - l2.length) - 1)  + "|\n")
    logOutput("|-" + l3 + " " * (((total_width ) - l3.length) - 1)  + "|\n")
    logOutput("-" * (total_width + 2) + "\n")
  }

}
