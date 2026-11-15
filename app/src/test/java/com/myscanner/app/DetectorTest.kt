package com.myscanner.app

import com.myscanner.app.scan.DocQuad
import com.myscanner.app.scan.DocumentDetector
import com.myscanner.app.scan.PageNet
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.imgcodecs.Imgcodecs
import org.robolectric.RobolectricTestRunner
import java.io.File
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min

/**
 * Runs the app's real page detector (Kotlin + both networks) on 196 labelled test images,
 * 35 of them hand-labelled real photos, and checks it against the reference results.
 */
@RunWith(RobolectricTestRunner::class)
class DetectorTest {

    private class Case(val set: String, val file: File, val gt: List<Pair<Double, Double>>, val reference: Double)

    private fun cases(dir: File): List<Case> = File(dir, "detect").listFiles { f -> f.name.endsWith(".png") }!!.sortedBy { it.name }.map { png ->
        val lines = File(png.path.removeSuffix(".png") + ".txt").readLines()
        val v = lines[0].trim().split(" ").map { it.toDouble() }
        Case(png.name.substringBeforeLast("_"), png, (0 until 4).map { v[it * 2] to v[it * 2 + 1] }, lines[1].toDouble())
    }

    /** Biggest corner error as a fraction of the page diagonal (corner order may start anywhere). */
    private fun error(q: List<Pair<Double, Double>>?, gt: List<Pair<Double, Double>>, w: Int, h: Int): Double {
        if (q == null) return 1.0
        val diag = hypot((gt[2].first - gt[0].first) * w, (gt[2].second - gt[0].second) * h)
        var best = 9.0
        for (g in listOf(gt, gt.reversed())) for (k in 0 until 4) {
            var worst = 0.0
            for (j in 0 until 4) {
                val p = g[(j + k) % 4]
                worst = max(worst, hypot((q[j].first - p.first) * w, (q[j].second - p.second) * h))
            }
            best = min(best, worst / diag)
        }
        return best
    }

    @Test
    fun traceOneImage() {
        TestSupport.requireOpenCv()
        val dir = TestSupport.requireData()
        for (name in (System.getenv("TRACE_IMAGES") ?: "").split(",").filter { it.isNotBlank() }) {
            val base0 = Imgcodecs.imread(File(dir, "detect/$name.png").path)
            val img = variant(base0, System.getenv("TRACE_VARIANT") ?: "original")
            DocumentDetector.trace = true
            DocumentDetector.traceDir = System.getenv("TRACE_DIR")
            DocumentDetector.detectWithConfidence(img)
            DocumentDetector.trace = false
            DocumentDetector.traceDir = null
            println("DETECTOR trace $name ${img.cols()}x${img.rows()}: ${DocumentDetector.lastDebug}\n${DocumentDetector.lastTrace}")
            img.release()
        }
    }

    private fun variant(base: org.opencv.core.Mat, v: String): org.opencv.core.Mat {
        val img = org.opencv.core.Mat()
        when (v) {
            "jpeg80" -> {
                val buf = org.opencv.core.MatOfByte()
                Imgcodecs.imencode(".jpg", base, buf, org.opencv.core.MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, 80))
                Imgcodecs.imdecode(buf, Imgcodecs.IMREAD_COLOR).copyTo(img); buf.release()
            }
            "noise" -> {
                org.opencv.core.Core.setRNGSeed(7)
                val noise = org.opencv.core.Mat(base.size(), org.opencv.core.CvType.CV_16SC3)
                org.opencv.core.Core.randn(noise, 0.0, 3.0)
                val wide = org.opencv.core.Mat(); base.convertTo(wide, org.opencv.core.CvType.CV_16SC3)
                org.opencv.core.Core.add(wide, noise, wide); wide.convertTo(img, org.opencv.core.CvType.CV_8UC3)
                noise.release(); wide.release()
            }
            "soft" -> org.opencv.imgproc.Imgproc.GaussianBlur(base, img, org.opencv.core.Size(0.0, 0.0), 0.8)
            "darker" -> base.convertTo(img, -1, 0.85, -4.0)
            "small" -> org.opencv.imgproc.Imgproc.resize(base, img, org.opencv.core.Size(), 0.625, 0.625, org.opencv.imgproc.Imgproc.INTER_AREA)
            else -> base.copyTo(img)
        }
        return img
    }

    /** Where the time goes (this PC is several times faster than a budget phone; the ratios are what matter). */
    @Test
    fun timing() {
        TestSupport.requireOpenCv()
        val dir = TestSupport.requireData()
        val files = cases(dir).filter { it.set == "real" || it.set == "amz_sheet" }.take(20)
        for (size in listOf(640, 400)) {
            var tDq = 0L; var tPn = 0L; var tEdges = 0L; var tAll = 0L
            repeat(2) { round ->
                for (c in files) {
                    val full = Imgcodecs.imread(c.file.path)
                    val img = org.opencv.core.Mat()
                    val s = size.toDouble() / max(full.cols(), full.rows())
                    org.opencv.imgproc.Imgproc.resize(full, img, org.opencv.core.Size(), s, s, org.opencv.imgproc.Imgproc.INTER_AREA)
                    val t0 = System.nanoTime(); DocQuad.run(img)
                    val t1 = System.nanoTime(); PageNet.predict(img)?.release()
                    val t2 = System.nanoTime(); DocumentDetector.detectWithConfidence(img, useNet = false)
                    val t3 = System.nanoTime(); DocumentDetector.detectWithConfidence(img, useNet = true)
                    val t4 = System.nanoTime()
                    if (round == 1) { tDq += t1 - t0; tPn += t2 - t1; tEdges += t3 - t2; tAll += t4 - t3 }
                    full.release(); img.release()
                }
            }
            val n = files.size * 1_000_000L
            println("PERF detector at $size px: corner network ${tDq / n} ms, page network ${tPn / n} ms, edge search ${tEdges / n} ms, everything ${tAll / n} ms")
        }
    }

    /** Writes, for every photo and variant, both candidate outlines' errors and the decision numbers. */
    @Test
    fun dumpChoiceFeatures() {
        TestSupport.requireOpenCv()
        val dir = TestSupport.requireData()
        val out = System.getenv("TRACE_DIR") ?: return
        val sb = StringBuilder("set,name,variant,e_classic,e_net,dq,best,iou,net_classic,net_side_score,net_dq\n")
        DocumentDetector.trace = true
        for (c in cases(dir)) {
            val base = Imgcodecs.imread(c.file.path)
            for (v in listOf("original", "jpeg80", "noise", "soft", "darker", "small")) {
                val img = variant(base, v)
                DocumentDetector.lastOptions = emptyList()
                DocumentDetector.detectWithConfidence(img)
                val o = DocumentDetector.lastOptions
                fun err(sx: String?): Double {
                    if (sx.isNullOrBlank()) return 1.0
                    val f = sx.trim().split(" ").map { it.toDouble() }
                    return error((0 until 4).map { f[it * 2] to f[it * 2 + 1] }, c.gt, img.cols(), img.rows())
                }
                sb.append("${c.set},${c.file.name},$v,%.4f,%.4f,${(o.getOrNull(2) ?: "-1 0 -1 -1 0 -1").replace(' ', ',')}\n".format(err(o.getOrNull(0)), err(o.getOrNull(1))))
                img.release()
            }
            base.release()
        }
        DocumentDetector.trace = false
        File(out, "features.csv").writeText(sb.toString())
        println("DETECTOR features written")
    }

    /** The same photos with small camera-like changes: the result must not depend on luck. */
    @Test
    fun detectorIsStableUnderSmallChanges() {
        TestSupport.requireOpenCv()
        val dir = TestSupport.requireData()
        val all = cases(dir).filter { it.set != "pack_out" && it.set != "amz_book" }
        val variants = listOf("original", "jpeg80", "noise", "soft", "darker")
        val fails = IntArray(variants.size); var flips = 0; var n = 0
        val perSet = HashMap<String, IntArray>()
        for (c in all) {
            val base = Imgcodecs.imread(c.file.path)
            var firstError = 0.0
            variants.forEachIndexed { vi, v ->
                val img = variant(base, v)
                val r = DocumentDetector.detectWithConfidence(img)
                val e = error(r?.quad?.points?.map { it.x.toDouble() to it.y.toDouble() }, c.gt, img.cols(), img.rows())
                img.release()
                if (e > 0.10) { fails[vi]++; perSet.getOrPut(c.set) { IntArray(variants.size) }[vi]++ }
                if (vi == 0) firstError = e else if ((e > 0.10) != (firstError > 0.10)) flips++
            }
            base.release(); n++
        }
        println("DETECTOR stability on $n photos, big failures: " + variants.indices.joinToString { "${variants[it]} ${fails[it]}" } + "; results that changed with the variant: $flips of ${n * (variants.size - 1)}")
        println("DETECTOR stability per set: " + perSet.keys.sorted().joinToString { k -> "$k ${perSet[k]!!.joinToString("/")}" })
    }

    @Test
    fun detectorMatchesReferenceOnTestPhotos() {
        TestSupport.requireOpenCv()
        val dir = TestSupport.requireData()
        val app = androidx.test.core.app.ApplicationProvider.getApplicationContext<android.content.Context>()
        PageNet.init(app); DocQuad.init(app)
        assertTrue("networks loaded", PageNet.available && DocQuad.available)
        val big = HashMap<String, Int>(); val total = HashMap<String, Int>(); val refBig = HashMap<String, Int>()
        var time = 0L; var n = 0; var unsure = 0
        for (c in cases(dir)) {
            val img = Imgcodecs.imread(c.file.path)
            val t0 = System.nanoTime()
            val r = DocumentDetector.detectWithConfidence(img)
            time += System.nanoTime() - t0; n++
            val q = r?.quad?.points?.map { it.x.toDouble() to it.y.toDouble() }
            val e = error(q, c.gt, img.cols(), img.rows())
            img.release()
            total.merge(c.set, 1, Int::plus)
            if (e > 0.10) big.merge(c.set, 1, Int::plus)
            if (c.reference > 0.10) refBig.merge(c.set, 1, Int::plus)
            if (r != null && !r.sure) unsure++
            if ((e > 0.10) != (c.reference > 0.10)) println("DETECTOR differs: ${c.file.name} kotlin %.3f reference %.3f | kotlin: ${DocumentDetector.lastDebug} | python pk,s_v3,iou,net_v3,rule: ${File(c.file.path.removeSuffix(".png") + ".txt").readLines().getOrNull(3)}".format(e, c.reference))
        }
        println("DETECTOR big failures (corner >10% off) per set: " + total.keys.sorted().joinToString { "$it ${big[it] ?: 0}/${total[it]} (ref ${refBig[it] ?: 0})" })
        println("DETECTOR avg ${time / n / 1_000_000} ms per 640px photo; flagged unsure: $unsure/$n")
        val real = listOf("real", "amz_sheet", "pack_in").sumOf { big[it] ?: 0 }
        val syn = listOf("standard", "unseen", "hard").sumOf { big[it] ?: 0 }
        assertTrue("real-photo big failures $real should be <= 6 of 35", real <= 6)
        assertTrue("synthetic big failures $syn should be <= 8 of 136", syn <= 8)
    }
}
