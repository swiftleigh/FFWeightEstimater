import org.datavec.api.records.reader.impl.csv.CSVRecordReader
import org.datavec.api.split.FileSplit
import org.deeplearning4j.datasets.datavec.RecordReaderDataSetIterator
import org.deeplearning4j.nn.api.OptimizationAlgorithm
import org.deeplearning4j.nn.conf.layers.{DenseLayer, OutputLayer}
import org.deeplearning4j.nn.conf.{NeuralNetConfiguration, Updater}
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork
import org.deeplearning4j.nn.weights.WeightInit
import org.deeplearning4j.ui.api.UIServer
import org.deeplearning4j.ui.stats.StatsListener
import org.deeplearning4j.ui.storage.{FileStatsStorage}
import org.deeplearning4j.util.ModelSerializer
import org.nd4j.linalg.api.ndarray.INDArray
import org.nd4j.linalg.dataset.DataSet
import org.nd4j.linalg.lossfunctions.LossFunctions

import java.io.File

object FFDimensionEstimator {

  val dataPath = "data/"
  val modelsPath = "models/"
  val statsPath = "stats/"
  var features: INDArray = null
  var labels: INDArray = null
  var config: CLArgs = null
  var model: MultiLayerNetwork = null

  lazy val modelFileName = config.species + "_" + config.feature + "_nn_model.zip"
  lazy val dataFileName = config.species + ".csv"
  lazy val statsFileName = config.species + ".dat"

  val colVals = Map("length" -> (Array(1, 2), Array(0)), "girth" -> (Array(0, 2), Array(1)), "weight" -> (Array(0, 1), Array(2)))

  case class CLArgs(species: String = "", // required
                    feature: String = "",
                    train: Boolean = false,
                    predict: Boolean = false,
                    loadModel: Boolean = false,
                    iterations: Int = 100)

  // Define Command Line Paramaters
  val parser = new scopt.OptionParser[CLArgs]("ff_weight_estimator") {

    head("ffweightestimator", "1.0")
    opt[String]('s', "species").required().action((x, c) =>
      c.copy(species = x)).text("Species is a string property")
    opt[Unit]('t', "train").action((x, c) =>
      c.copy(train = true)).text("Flag to run training on dataset")
    opt[Unit]('p', "predict").action((x, c) =>
      c.copy(predict = true)).text("Flag to run prediction on dataset")
    opt[Unit]('l', "load existing model").action((x, c) =>
      c.copy(loadModel = true)).text("iterations is an integer property")
    opt[Int]('i', "iterations").action((x, c) =>
      c.copy(iterations = x)).text("iterations is an integer property")
    opt[String]('f', "feature to train").required().action((x, c) =>
      c.copy(feature = x)).text("Flag to run training on data set")

  }

  /**
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {

    parser.parse(args, CLArgs()) match {
      case Some(config) =>
        run(config)

      case None =>
        throw new Exception()
    }

  }


  /**
    *
    * @param c
    */
  def run(c: CLArgs): Unit = {

    config = c

    model = if (config.loadModel) loadModel() else createModel()

    val (features, labels) = loadData(dataPath + dataFileName, colVals(config.feature)._1, colVals(config.feature)._2)
    this.features = features
    this.labels = labels

    attachStorage(model)

    if (config.train) {
      model.fit(new DataSet(features, labels))
      saveModel()
    }

    if (config.predict)
      print(model.output(features))

  }

  /**
    *
    * @param path
    * @param fCols
    * @param lCols
    * @return
    */
  def loadData(path: String, fCols: Array[Int], lCols: Array[Int]): (INDArray, INDArray) = {

    val recordReader = new CSVRecordReader(0, ",")
    recordReader.initialize(new FileSplit(new java.io.File(path)))

    val iterator = new RecordReaderDataSetIterator(recordReader, 3000)
    val dataSet = iterator.next
    val features = dataSet.getFeatures
    (features.getColumns(fCols: _*), features.getColumns(lCols: _*))

  }

  /***
    *
    * @param network
    */
  def attachStorage(network: MultiLayerNetwork): Unit = {

    val uiServer = UIServer.getInstance
    val statsStorage = new FileStatsStorage(new File(statsPath + statsFileName))
    uiServer.attach(statsStorage)
    model.setListeners(new StatsListener(statsStorage))

  }

  /**
    *
    */
  def saveModel(): Unit = ModelSerializer.writeModel(model, modelsPath + modelFileName, false)


  /**
    *
    * @return
    */
  def loadModel(): MultiLayerNetwork =  ModelSerializer.restoreMultiLayerNetwork(modelsPath + modelFileName, false)


  /**
    *
    * @return
    */
  def createModel(): MultiLayerNetwork = {

    val conf = new NeuralNetConfiguration.Builder()
      .seed(1334)
      .iterations(2000)
      .optimizationAlgo(OptimizationAlgorithm.LINE_GRADIENT_DESCENT)
      .learningRate(0.01)
      .weightInit(WeightInit.XAVIER)
      .updater(Updater.NESTEROVS)
      .list()
      .layer(0, new DenseLayer.Builder().nIn(2).nOut(8)
        .activation("LeakyReLU")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(1, new DenseLayer.Builder().nIn(8).nOut(8)
        .activation("LeakyReLU")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(2, new DenseLayer.Builder().nIn(8).nOut(8)
        .activation("LeakyReLU")
        .weightInit(WeightInit.XAVIER)
        .build())
      .layer(3, new OutputLayer.Builder(LossFunctions.LossFunction.SQUARED_LOSS)
        .activation("identity")
        .nIn(8).nOut(1).build())
      .backprop(true)
      .pretrain(false)
      .build()

    new MultiLayerNetwork(conf)
  }

}
