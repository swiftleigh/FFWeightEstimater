
import java.io.File

import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers._
import org.deeplearning4j.nn.conf.{LearningRatePolicy, NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import java.lang.Double
import java.net.URI
import java.util

import org.datavec.api.split.FileSplit
import org.datavec.image.recordreader.objdetect.{ImageObject, ImageObjectLabelProvider, ObjectDetectionRecordReader}
import org.deeplearning4j.nn.conf.layers.objdetect.Yolo2OutputLayer
import org.nd4j.linalg.activations.Activation

import collection.JavaConversions._

trait FFLabelProvider extends ImageObjectLabelProvider {
  override def getImageObjectsForPath(path: String): util.List[ImageObject] = ???

  override def getImageObjectsForPath(uri: URI): util.List[ImageObject] = ???
}

object FFLabelProvider {

  def apply(path: String) {

  }

}

object FFYolo {

  val dataPath = "data/"
  val modelsPath = "models/"
  val statsPath = "stats/"
  val modelFileName = ""

  var model: MultiLayerNetwork = null

  val nfeatures = 1 //ds.getFeatures.getRow(0).length // hyper, hyper parameter
  val numRows = Math.sqrt(nfeatures).toInt // numRows * numColumns must equal columns in initial data * channels
  val numColumns = Math.sqrt(nfeatures).toInt // numRows * numColumns must equal columns in initial data * channels
  val nChannels = 3 // would be 3 if color image w R,G,B
  val outputNum = 2 // # of classes (# of columns in output)
  val iterations = 1
  val seed = 1234
  val listenerFreq = 1
  val nepochs = 20
  val nbatch = 32 // recommended between 16 and 128

  val lrSchedule: java.util.Map[Integer, Double] = Map(new Integer(0) -> new Double(0.01), new Integer(1000) -> new Double(0.005), new Integer(3000) -> new Double(0.001))


  def main(args: Array[String]): Unit = {
    model = createModel()
    train()
  }

  def train(): Unit = {

    val labelProvider = new VocLabelProvider("/Users/lwilliams/Work/FishFace/tensorflow/project/largemouth_tensor/")
    val reader = new ObjectDetectionRecordReader(416, 416, 3, 4, 4, labelProvider)
    val fs = new FileSplit(new File("/Users/lwilliams/Work/FishFace/tensorflow/project/largemouth_tensor/images"))
    val objd = reader.initialize(fs)

    reader.next()

  }

  def createModel(): MultiLayerNetwork = {

    val conf = new NeuralNetConfiguration.Builder()
      .seed(1334)
      .miniBatch(true)
      .optimizationAlgo(OptimizationAlgorithm.STOCHASTIC_GRADIENT_DESCENT)
      .learningRate(0.001)
      .learningRateSchedule(lrSchedule)
      .learningRateDecayPolicy(LearningRatePolicy.Schedule)
      .momentum(0.9)
      .list()
      .layer(0, new ConvolutionLayer.Builder(3, 3)
        .padding(1, 1)
        .nIn(nChannels)
        .stride(1, 1) // default stride(2,2)
        .nOut(16)
        .activation(Activation.IDENTITY)
        .build())
      .layer(1, new BatchNormalization.Builder()
        .build())
      .layer(2, new ActivationLayer.Builder()
        .activation(Activation.RELU)
        .build())
      .layer(3, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
        .build())
      .layer(4, new ConvolutionLayer.Builder(3, 3)
        .padding(1, 1)
        .nIn(nChannels)
        .stride(1, 1) // default stride(2,2)
        .nOut(32) // # of feature maps
        .build())
      .layer(5, new BatchNormalization.Builder()
        .build())
      .layer(6, new ActivationLayer.Builder()
        .activation(Activation.RELU)
        .build())
      .layer(7, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
        .build())
      .layer(8, new ConvolutionLayer.Builder(3, 3)
        .padding(1, 1)
        .nIn(nChannels)
        .stride(1, 1) // default stride(2,2)
        .nOut(64) // # of feature maps
        .build())
      .layer(9, new BatchNormalization.Builder()
        .build())
      .layer(10, new ActivationLayer.Builder()
        .activation(Activation.RELU)
        .build())
      .layer(11, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2))
        .build())
      .layer(12, new ConvolutionLayer.Builder(3, 3)
        .padding(1, 1)
        .nIn(nChannels)
        .stride(1, 1) // default stride(2,2)
        .nOut(128) // # of feature maps
        .build())
      .layer(13, new BatchNormalization.Builder().build())
      .layer(14, new ActivationLayer.Builder()
        .activation(Activation.RELU)
        .build())
      .layer(15, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2)).build())
      .layer(16, new ConvolutionLayer.Builder(3, 3)
        .padding(1, 1)
        .nIn(nChannels)
        .stride(1, 1) // default stride(2,2)
        .nOut(256) // # of feature maps
        .build())
      .layer(17, new BatchNormalization.Builder().build())
      .layer(18, new ActivationLayer.Builder()
        .activation(Activation.RELU)
        .build())
      .layer(19, new SubsamplingLayer.Builder(SubsamplingLayer.PoolingType.MAX, Array(2, 2)).build())
      .layer(20, new ConvolutionLayer.Builder(3, 3)
        .padding(1, 1)
        .nIn(nChannels)
        .stride(1, 1) // default stride(2,2)
        .nOut(512) // # of feature maps
        .build())
      .layer(21, new BatchNormalization.Builder().build())
      .layer(22, new ActivationLayer.Builder()
        .activation(Activation.RELU)
        .build())
      .layer(23, new ConvolutionLayer.Builder(3, 3)
        .padding(1, 1)
        .nIn(nChannels)
        .stride(1, 1) // default stride(2,2)
        .nOut(1024) // # of feature maps
        .build())
      .layer(24, new BatchNormalization.Builder().build())
      .layer(25, new ActivationLayer.Builder()
        .activation(Activation.RELU)
        .build())
      .layer(26, new ConvolutionLayer.Builder(3, 3)
        .padding(1, 1)
        .nIn(nChannels)
        .stride(1, 1) // default stride(2,2)
        .nOut(1024) // # of feature maps
        .build())
      .layer(27, new BatchNormalization.Builder().build())
      .layer(28, new ActivationLayer.Builder()
        .activation(Activation.RELU)
        .build())
      .layer(29, new ConvolutionLayer.Builder(1, 1)
        .padding(1, 1)
        .nIn(nChannels)
        .stride(1, 1)
        .nOut(125) // # of feature maps
        .build())
      .layer(31, new Yolo2OutputLayer.Builder()
        .build())
      .backprop(true)
      .pretrain(false)
      .build()

    new MultiLayerNetwork(conf)
  }

}