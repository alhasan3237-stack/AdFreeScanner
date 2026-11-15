package com.myscanner.app

import androidx.test.core.app.ApplicationProvider
import com.myscanner.app.scan.DocQuad
import com.myscanner.app.scan.DocumentDetector
import com.myscanner.app.scan.ImageProcessor
import com.myscanner.app.scan.PageNet
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import org.robolectric.RobolectricTestRunner
import java.io.File
import kotlin.concurrent.thread

/** The live camera path: small frames, reused corner hints, while photos are processed on another thread. */
@RunWith(RobolectricTestRunner::class)
class LiveFinderTest {
    @Test
    fun liveFramesWithPhotoWorkAtTheSameTime() {
        TestSupport.requireOpenCv()
        val dir = TestSupport.requireData()
        val app = ApplicationProvider.getApplicationContext<android.content.Context>()
        PageNet.init(app); DocQuad.init(app)
        val photos = (1..7).map { File(dir, "photos/r0$it.jpg") }.filter { it.exists() }
        val frames = photos.map { f ->
            val big = Imgcodecs.imread(f.path); val s = 640.0 / maxOf(big.cols(), big.rows())
            Mat().also { Imgproc.resize(big, it, Size(big.cols() * s, big.rows() * s)); big.release() }
        }
        var stop = false
        val other = thread { while (!stop) photos.forEach { println("PHOTO " + ImageProcessor.detect(it).second) } }
        var hint: DocQuad.Result? = null
        repeat(60) { i ->
            val frame = frames[(i / 3) % frames.size]
            val small = Mat(); Imgproc.resize(frame, small, Size(400.0, 300.0), 0.0, 0.0, Imgproc.INTER_AREA)
            ImageProcessor.rotateInPlace(small, 90)
            if (i % 4 == 0) hint = DocQuad.run(small)
            val q = DocumentDetector.detect(small, hint)
            println("LIVE $i ${q?.points?.joinToString { "%.2f,%.2f".format(it.x, it.y) }}")
            small.release()
        }
        stop = true; other.join()
    }
}
