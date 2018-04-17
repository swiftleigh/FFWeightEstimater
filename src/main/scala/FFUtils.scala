import java.io.File

object FFUtils {
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
}
