import sbt.url

name := "DSmartML"
version := "0.1"

scalaVersion := "2.11.8"

libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.3.0",
  "org.apache.spark" % "spark-sql_2.11" % "2.3.0",
  "org.apache.spark" % "spark-streaming_2.11" % "2.3.0",
  "org.apache.spark" % "spark-mllib_2.11" % "2.3.0")


retrieveManaged := true
mainClass in (Compile, run) := Some("org.dsmartml.Run_Main")



