import java.io.File
import com.sksamuel.scrimage.Image
import com.sksamuel.scrimage.nio.JpegWriter
import scala.util.Random

object FFImageMaker {

  var config:CLArgs = null

  case class CLArgs(sourceDir: String = "", // required
                    destDir: String = "",
                    width: Int = 0,
                    height: Int = 0)

  // Define Command Line Paramaters
  val parser = new scopt.OptionParser[CLArgs]("ff_weight_estimator") {

    head("ffweightestimator", "1.0")
    opt[String]('s', "source directory").required().action((x, c) =>
      c.copy(sourceDir = x)).text("")
    opt[String]('d', "destination directory").required().action((x, c) =>
      c.copy(destDir = x)).text("")
    opt[Int]('w', "width").required().action((x, c) =>
      c.copy(width = x)).text("")
    opt[Int]('h', "height").action((x, c) =>
      c.copy(height = x)).required().text("")

  }

  /**
    *
    * @param c
    */
  def run(c: CLArgs):Unit = {
    config = c

    val files = listOfFiles(config.sourceDir + "images", "jpg")

    for(file <- files) {
      val image = Image.fromFile(file).fit(444, 444)
      implicit val writer = JpegWriter()
      image.output(new File(config.destDir + file.getName()))
    }

  }

  /***
    *
    * @param dir
    * @param ext
    * @return
    */
  def listOfFiles(dir: String, ext: String): List[File] = {
    val d = new File(dir)
    if (d.exists && d.isDirectory) {
      d.listFiles.filter(f => f.isFile && f.getName.contains(ext)).toList
    } else {
      List[File]()
    }
  }

  /**
    *
    * @param args
    */
  def main(args: Array[String]): Unit = {

    parser.parse(args, CLArgs()) match {
      case Some(c) =>
        run(c)

      case None =>
        throw new Exception()
    }

  }

}
