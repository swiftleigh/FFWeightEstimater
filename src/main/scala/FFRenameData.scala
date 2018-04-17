import java.io.File
import java.nio.file.Files.copy
import java.nio.file.{Path, Paths}

import FFUtils._

object FFRenameData {


  case class CLArgs(sourceDir: String = "", destDir: String = "")

  val parser = new scopt.OptionParser[CLArgs]("ff_weight_estimator") {

    head("ffweightestimator", "1.0")
    opt[String]('s', "source directory").required().action((x, c) =>
      c.copy(sourceDir = x)).text("")
    opt[String]('d', "destination directory").required().action((x, c) =>
      c.copy(destDir = x)).text("")

  }

  def main(args: Array[String]): Unit = {

    parser.parse(args, CLArgs()) match {
      case Some(c) =>
        run(c)

      case None =>
        throw new Exception("Invalid Command")
    }
  }

  def run(config: CLArgs): Unit = {

    val imageFiles = listOfFiles(config.sourceDir + "/images", "jpg")

    for ((imageFile, index) <- imageFiles.view.zipWithIndex) {

      val labelFile = new File(config.sourceDir + "/labels/" + imageFile.getName.replace(".jpg", ".xml"))

      if (labelFile.exists()) {
        val fileName = Array.fill(10 - index.toString.length) {"0"}.mkString + index.toString
        copy(labelFile.toPath, Paths.get(config.destDir + "/labels/" + fileName + ".xml"))
        copy(imageFile.toPath, Paths.get(config.destDir + "/images/" + fileName + ".jpg"))
      }

    }
  }


}
