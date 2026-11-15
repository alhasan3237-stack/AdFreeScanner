package com.myscanner.app

import com.myscanner.app.data.ScanFilter
import com.myscanner.app.scan.ImageProcessor
import com.myscanner.app.scan.Quad
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Core
import org.opencv.core.Mat
import org.opencv.imgcodecs.Imgcodecs
import org.robolectric.RobolectricTestRunner
import java.io.File

/**
 * Runs the real photo pipeline (decode, find page, straighten, filters) on full-size phone photos.
 * Prints timings, and compares every filter's output with saved reference images, so speed-ups
 * can't quietly change how pages look.
 */
@RunWith(RobolectricTestRunner::class)
class ImagePipelineTest {

    private fun photos(dir: File) = File(dir, "photos").listFiles { f -> f.name.endsWith(".jpg") }!!.sortedBy { it.name }

    private inline fun <T> timed(sink: HashMap<String, Long>, key: String, block: () -> T): T {
        val t = System.nanoTime(); val r = block(); sink.merge(key, System.nanoTime() - t, Long::plus); return r
    }

    @Test
    fun pipelineTimingAndLook() {
        TestSupport.requireOpenCv()
        val dir = TestSupport.requireData()
        val refDir = File(dir, "filter-reference").apply { mkdirs() }
        val record = System.getenv("RECORD_REFERENCE") == "1"
        val t = HashMap<String, Long>()
        var worstMean = 0.0; var worstName = ""
        val list = photos(dir).filter { it.name in setOf("r02.jpg", "r03.jpg", "r04.jpg", "r06.jpg") }
        for (photo in list) {
            val (quad, _) = timed(t, "find page") { ImageProcessor.detect(photo) }
            // Preview size (what the review screen shows)
            val page = timed(t, "preview: decode+straighten") { ImageProcessor.warpPage(photo, quad, 1600) }
            val flat = timed(t, "preview: remove shadows") { ImageProcessor.flatten(page) }
            for (f in ScanFilter.entries) {
                val out = timed(t, "preview: filter ${f.name}") { ImageProcessor.applyFilter(page, f, flat) }
                val ref = File(refDir, "${photo.nameWithoutExtension}_${f.name}.png")
                if (record || !ref.exists()) Imgcodecs.imwrite(ref.path, out)
                else {
                    val expected = Imgcodecs.imread(ref.path, if (out.channels() == 1) Imgcodecs.IMREAD_GRAYSCALE else Imgcodecs.IMREAD_COLOR)
                    if (expected.size() == out.size()) {
                        val d = Mat(); Core.absdiff(out, expected, d)
                        val mean = Core.mean(d).`val`.take(out.channels()).average()
                        if (mean > worstMean) { worstMean = mean; worstName = ref.name }
                        d.release()
                    } else { worstMean = 999.0; worstName = ref.name + " (size changed)" }
                    expected.release()
                }
                out.release()
            }
            page.release(); flat.release()
            // Full size (what gets saved)
            val full = timed(t, "save: whole page (AUTO)") { ImageProcessor.render(photo, quad, ScanFilter.AUTO, 0, 4000) }
            val tmp = File.createTempFile("page", ".jpg")
            timed(t, "save: write JPEG") { ImageProcessor.saveJpeg(full, tmp) }
            full.release(); tmp.delete()
            timed(t, "blur check") { ImageProcessor.sharpness(photo, quad) }
        }
        println("PERF pipeline, average per photo over ${list.size} photos:")
        t.keys.sorted().forEach { k -> println("PERF   %-32s %5d ms".format(k, t[k]!! / list.size / 1_000_000)) }
        println("FILTER look vs reference: worst mean difference %.2f (of 255) in %s".format(worstMean, worstName))
        if (!record) assertTrue("filters changed visibly: $worstMean in $worstName", worstMean < 2.5)
    }
}
