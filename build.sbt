name := "FFWeightEstimator"

version := "0.1"

scalaVersion := "2.11.8"


libraryDependencies ++= Seq(
  "org.apache.spark" % "spark-core_2.11" % "2.2.0",
  "org.apache.spark" % "spark-mllib_2.11" % "2.2.0",
  "org.deeplearning4j" % "deeplearning4j-core" % "0.9.1",
  "org.deeplearning4j" % "dl4j-spark_2.11" % "0.5.0",
  "org.nd4j" % "nd4j-native-platform" % "0.8.0",
  "org.deeplearning4j" % "deeplearning4j-scaleout-api" % "1.0",
  "org.nd4j" % "nd4s_2.11" % "0.8.0",
  "org.deeplearning4j" % "deeplearning4j-ui_2.11" % "0.7.2",
  "com.twelvemonkeys.imageio" % "imageio-core" % "3.1.1",
  "com.sksamuel.scrimage" % "scrimage-core_2.11" % "2.1.8",
  "com.github.scopt" %% "scopt" % "3.5.0"
)