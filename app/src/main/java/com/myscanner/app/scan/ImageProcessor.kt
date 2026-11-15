package com.myscanner.app.scan

import android.graphics.Bitmap
import android.graphics.BitmapFactory
import androidx.exifinterface.media.ExifInterface
import com.myscanner.app.data.ScanFilter
import org.opencv.android.Utils
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfFloat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.imgcodecs.Imgcodecs
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.abs
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * All image work: loading photos at the right size, straightening pages, and the filters.
 * Images are kept in OpenCV's native memory (not the Java heap), which keeps budget phones
 * from running out of memory.
 */
object ImageProcessor {

    /** Fallback biggest page side (the session's photo quality normally decides). */
    const val MAX_SIDE = 3000

    // ---------- Loading ----------

    private fun exifDegrees(file: File): Int = try {
        ExifInterface(file.path).rotationDegrees
    } catch (e: Exception) {
        0
    }

    private fun imageSize(file: File): Pair<Int, Int> {
        val o = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, o)
        return o.outWidth to o.outHeight
    }

    /** A photo decoded at 1/2, 1/4 or 1/8 size, kept for a moment so the next step can reuse it. */
    private class Decoded(val key: String, val reduce: Int, val gray: Boolean, val mat: Mat)

    private val decodedCache = ArrayList<Decoded>()
    private const val DECODED_KEPT = 2

    /** Frees the photos kept in memory by [load] (call when a scan ends or memory is low). */
    fun clearCache() = synchronized(decodedCache) {
        decodedCache.forEach { it.mat.release() }
        decodedCache.clear()
    }

    /**
     * Loads a photo upright, shrunk so its long side is at most [maxSide].
     * Decoding at reduced size (1/2, 1/4, 1/8) happens inside the JPEG decoder, which is fast, and
     * the last couple of reduced decodes are kept: right after a shot, finding the page, the
     * animation and the preview all read the same photo, and decoding it is the slow part.
     * With [soft], a decode up to 20% smaller than asked is accepted (for previews).
     */
    fun load(file: File, maxSide: Int, gray: Boolean = false, soft: Boolean = false): Mat {
        val (w, h) = imageSize(file)
        val longest = max(w, h)
        val need = if (soft) maxSide * 0.8 else maxSide.toDouble()
        var reduce = 1
        if (longest > 0) {
            while (reduce < 8 && longest / (reduce * 2) >= need) reduce *= 2
        }
        val key = file.path + "|" + file.length() + "|" + file.lastModified()
        var mat: Mat? = null
        if (reduce >= 2) synchronized(decodedCache) {
            // A finer (or equal) decode of the same photo is as good as a new one.
            decodedCache.filter { it.key == key && it.gray == gray && it.reduce <= reduce }
                .minByOrNull { -it.reduce }?.let { hit ->
                    decodedCache.remove(hit); decodedCache.add(hit) // most recently used last
                    mat = hit.mat.clone()
                }
        }
        if (mat == null) {
            val base = when (reduce) {
                2 -> if (gray) Imgcodecs.IMREAD_REDUCED_GRAYSCALE_2 else Imgcodecs.IMREAD_REDUCED_COLOR_2
                4 -> if (gray) Imgcodecs.IMREAD_REDUCED_GRAYSCALE_4 else Imgcodecs.IMREAD_REDUCED_COLOR_4
                8 -> if (gray) Imgcodecs.IMREAD_REDUCED_GRAYSCALE_8 else Imgcodecs.IMREAD_REDUCED_COLOR_8
                else -> if (gray) Imgcodecs.IMREAD_GRAYSCALE else Imgcodecs.IMREAD_COLOR
            }
            val read = Imgcodecs.imread(file.path, base or Imgcodecs.IMREAD_IGNORE_ORIENTATION)
            if (read.empty()) {
                read.release()
                error("Could not read the image")
            }
            if (reduce >= 2) synchronized(decodedCache) {
                decodedCache.removeAll { old -> (old.key == key && old.gray == gray && old.reduce >= reduce).also { if (it) old.mat.release() } }
                decodedCache.add(Decoded(key, reduce, gray, read.clone()))
                while (decodedCache.size > DECODED_KEPT) decodedCache.removeAt(0).mat.release()
            }
            mat = read
        }
        val out = mat!!
        resizeToFit(out, maxSide)
        rotateInPlace(out, exifDegrees(file))
        return out
    }

    /** Shrinks in place so the long side is at most [maxSide]. */
    fun resizeToFit(mat: Mat, maxSide: Int) {
        val longest = max(mat.cols(), mat.rows())
        if (maxSide in 1 until longest) {
            val s = maxSide.toDouble() / longest
            Imgproc.resize(mat, mat, Size(mat.cols() * s, mat.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
        }
    }

    /** Rotates clockwise by 0, 90, 180 or 270 degrees, in place. */
    fun rotateInPlace(mat: Mat, degrees: Int) {
        when (((degrees % 360) + 360) % 360) {
            90 -> Core.rotate(mat, mat, Core.ROTATE_90_CLOCKWISE)
            180 -> Core.rotate(mat, mat, Core.ROTATE_180)
            270 -> Core.rotate(mat, mat, Core.ROTATE_90_COUNTERCLOCKWISE)
        }
    }

    /**
     * Finds the page in a saved photo (in colour, which is more accurate).
     * Returns the outline and whether the detector is sure; falls back to the whole photo (not sure).
     */
    fun detect(file: File): Pair<Quad, Boolean> {
        val img = load(file, 640)
        try {
            val r = DocumentDetector.detectWithConfidence(img)
            return if (r != null) r.quad to r.sure else Quad.FULL to false
        } finally {
            img.release()
        }
    }

    // ---------- Straightening ----------

    /** Cuts out the page and flattens it into a straight rectangle (fixes tilt and perspective). */
    fun warp(src: Mat, quad: Quad, interpolation: Int = Imgproc.INTER_CUBIC): Mat {
        val w = src.cols().toDouble()
        val h = src.rows().toDouble()
        val p = quad.points.map { Point(it.x * w, it.y * h) }
        val outW = max(dist(p[0], p[1]), dist(p[3], p[2])).roundToInt().coerceAtLeast(8)
        val outH = max(dist(p[0], p[3]), dist(p[1], p[2])).roundToInt().coerceAtLeast(8)

        val from = MatOfPoint2f(p[0], p[1], p[2], p[3])
        val to = MatOfPoint2f(
            Point(0.0, 0.0),
            Point(outW - 1.0, 0.0),
            Point(outW - 1.0, outH - 1.0),
            Point(0.0, outH - 1.0),
        )
        val m = Imgproc.getPerspectiveTransform(from, to)
        val out = Mat()
        // Cubic keeps letters crisp when straightening the saved page; previews use the quicker linear.
        Imgproc.warpPerspective(src, out, m, Size(outW.toDouble(), outH.toDouble()), interpolation, Core.BORDER_REPLICATE)
        from.release(); to.release(); m.release()
        return out
    }

    private fun dist(a: Point, b: Point) = hypot(a.x - b.x, a.y - b.y)

    // ---------- Filters ----------
    // Version 4, tuned on 24 realistic test photos (warm/green light, dim + noisy, hard and soft
    // shadows, glare, coloured paper, photos, black boxes, highlighter, pencil, receipts).
    //
    // Shadow removal finds the bare paper first (not text, not photos, not coloured marks, not dark
    // boxes), estimates the lighting from the paper only, and fills the gaps under everything else.
    // So shadows disappear while photos, boxes, coloured headers and highlighter keep their look.

    /** Returns a new image with the filter applied. [bgr] is a colour (3-channel) image. */
    fun applyFilter(bgr: Mat, filter: ScanFilter, flat: Mat? = null): Mat = when (filter) {
        ScanFilter.ORIGINAL -> bgr.clone()
        ScanFilter.AUTO -> withFlat(bgr, flat) { toneL(it, white = 0.92, blackPct = 0.5, gamma = 1.0, sharpenAmount = 0.9, chroma = 1.1) }
        ScanFilter.MAGIC -> withFlat(bgr, flat) { toneL(it, white = 0.88, blackPct = 1.5, gamma = 1.15, sharpenAmount = 1.0, chroma = 1.45) }
        ScanFilter.LIGHTEN -> lighten(bgr)
        ScanFilter.GRAY -> {
            val g = flatGray(bgr, flat)
            val lv = levels(g, blackPoint(g), 0.92 * 255, 1.0)
            g.release()
            sharpen(lv, 0.9).also { lv.release() }
        }
        ScanFilter.BW -> {
            val g = flatGray(bgr, flat)
            Imgproc.GaussianBlur(g, g, Size(3.0, 3.0), 0.0)
            val block = odd(max(15, max(g.cols(), g.rows()) / 50))
            val t = Mat()
            Imgproc.adaptiveThreshold(g, t, 255.0, Imgproc.ADAPTIVE_THRESH_GAUSSIAN_C, Imgproc.THRESH_BINARY, block, 15.0)
            val white = Mat()
            Imgproc.threshold(g, white, 215.0, 255.0, Imgproc.THRESH_BINARY)       // clean paper stays white
            Core.bitwise_or(t, white, t)
            val notDark = Mat()
            Imgproc.threshold(g, notDark, 89.0, 255.0, Imgproc.THRESH_BINARY)      // solid dark areas stay solid
            Core.bitwise_and(t, notDark, t)
            white.release(); notDark.release(); g.release()
            val out = Mat()
            Imgproc.medianBlur(t, out, 3)
            t.release()
            out
        }
        ScanFilter.BW_SOFT -> {
            val g = flatGray(bgr, flat)
            val out = levels(g, min(140.0, blackPoint(g) + 60), 205.0, 1.6)
            g.release()
            out
        }
    }

    /** Runs [block] on the shadow-free image: the one given (cached), or a fresh one. */
    private inline fun withFlat(bgr: Mat, given: Mat?, block: (Mat) -> Mat): Mat {
        val flat = given ?: flatten(bgr)
        return block(flat).also { if (given == null) flat.release() }
    }

    private fun flatGray(bgr: Mat, given: Mat?): Mat {
        val flat = given ?: flatten(bgr)
        val g = Mat()
        Imgproc.cvtColor(flat, g, Imgproc.COLOR_BGR2GRAY)
        if (given == null) flat.release()
        return g
    }

    /** Divides the photo by the paper's lighting: shadows and colour casts disappear. Returns 8-bit BGR. */
    fun flatten(bgr: Mat): Mat {
        val light = lightingMap(bgr) // small (at most WORK px), float, 3 channels
        val rows = bgr.rows(); val cols = bgr.cols()
        val out = Mat(rows, cols, CvType.CV_8UC3)
        // The lighting is enlarged and applied a strip at a time. A full-size float copy of a 12 MP page
        // would need ~300 MB; strips need a few MB, which is also faster (stays in the CPU cache).
        val sx = light.cols().toDouble() / cols; val sy = light.rows().toDouble() / rows
        val m = Mat(2, 3, CvType.CV_64F)
        val strip = Mat(); val f = Mat()
        var y0 = 0
        while (y0 < rows) {
            val y1 = min(rows, y0 + STRIP)
            // Same pixel mapping as resizing the whole map with bilinear interpolation.
            m.put(0, 0, sx, 0.0, 0.5 * sx - 0.5, 0.0, sy, (y0 + 0.5) * sy - 0.5)
            Imgproc.warpAffine(
                light, strip, m, Size(cols.toDouble(), (y1 - y0).toDouble()),
                Imgproc.INTER_LINEAR or Imgproc.WARP_INVERSE_MAP, Core.BORDER_REPLICATE, Scalar(0.0),
            )
            val src = bgr.submat(y0, y1, 0, cols)
            src.convertTo(f, CvType.CV_32F)
            src.release()
            Core.max(strip, Scalar(1.0, 1.0, 1.0), strip)
            Core.divide(f, strip, f, 255.0)
            val dst = out.submat(y0, y1, 0, cols)
            f.convertTo(dst, CvType.CV_8U) // saturates at 255
            dst.release()
            y0 = y1
        }
        m.release(); strip.release(); f.release(); light.release()
        return out
    }

    private const val WORK = 1400
    private const val STRIP = 256

    /** The paper's colour and brightness (float, 3 channels) at working size: at most [WORK] px. */
    private fun lightingMap(bgr: Mat): Mat {
        val longest = max(bgr.cols(), bgr.rows())
        val s = min(1.0, WORK.toDouble() / longest)
        val small = Mat()
        if (s < 1) Imgproc.resize(bgr, small, Size((bgr.cols() * s).roundToInt().toDouble(), (bgr.rows() * s).roundToInt().toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        else bgr.copyTo(small)
        val mask = paperMask(small)
        val sf = Mat()
        small.convertTo(sf, CvType.CV_32F)
        small.release()
        val bg = pushPull(sf, mask)
        sf.release(); mask.release()
        Imgproc.GaussianBlur(bg, bg, Size(0.0, 0.0), 1.2)
        return bg
    }

    /**
     * Gaussian blur with a big radius, done on a copy [shrink] times smaller and enlarged again.
     * For wide blurs the result is practically the same, at a small fraction of the time.
     */
    private fun wideBlur(src: Mat, dst: Mat, sigma: Double, shrink: Int) {
        val small = Mat()
        val sw = max(1, (src.cols() + shrink - 1) / shrink); val sh = max(1, (src.rows() + shrink - 1) / shrink)
        Imgproc.resize(src, small, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
        Imgproc.GaussianBlur(small, small, Size(0.0, 0.0), sigma / shrink)
        Imgproc.resize(small, dst, src.size(), 0.0, 0.0, Imgproc.INTER_LINEAR)
        small.release()
    }

    private fun gradMag(g: Mat, sigma: Double): Mat {
        val b = Mat(); val dx = Mat(); val dy = Mat(); val m = Mat()
        Imgproc.GaussianBlur(g, b, Size(0.0, 0.0), sigma)
        Imgproc.Sobel(b, dx, CvType.CV_32F, 1, 0, 3)
        Imgproc.Sobel(b, dy, CvType.CV_32F, 0, 1, 3)
        Core.magnitude(dx, dy, m)
        b.release(); dx.release(); dy.release()
        return m
    }

    /**
     * 1.0 where the pixel is bare paper (lit or shadowed), else 0.0. CV_32F, same size as [small].
     *  - sharp, clear edges are print (shadow edges are always softer) - excluded with a small margin,
     *  - pixels far darker than the paper nearby are dark print (boxes, photos) - excluded,
     *  - regions fenced in by print (table cells, highlighter, bars) are accepted only if they match
     *    the accepted paper right next to them, spreading outwards cell by cell.
     */
    private fun paperMask(small: Mat): Mat {
        val w = small.cols(); val h = small.rows(); val n = w * h
        val g8 = Mat(); Imgproc.cvtColor(small, g8, Imgproc.COLOR_BGR2GRAY)
        val g = Mat(); g8.convertTo(g, CvType.CV_32F); g8.release()

        val fine = gradMag(g, 0.7)
        val coarse = gradMag(g, 3.0)
        val localMean = Mat()
        wideBlur(g, localMean, 8.0, 4)
        val fineA = FloatArray(n); fine.get(0, 0, fineA)
        val coarseA = FloatArray(n); coarse.get(0, 0, coarseA)
        val meanA = FloatArray(n); localMean.get(0, 0, meanA)
        fine.release(); coarse.release(); localMean.release()

        val edgeB = ByteArray(n)
        for (i in 0 until n) {
            val rel = fineA[i] / (meanA[i] + 10f)
            val sharp = fineA[i] / (coarseA[i] + 1f)
            if (rel > 0.22f && sharp > 1.9f) edgeB[i] = 1
        }
        val edge = Mat(h, w, CvType.CV_8UC1); edge.put(0, 0, edgeB)
        val k5 = Imgproc.getStructuringElement(Imgproc.MORPH_ELLIPSE, Size(5.0, 5.0))
        Imgproc.dilate(edge, edge, k5); k5.release()
        edge.get(0, 0, edgeB); edge.release()
        // Bright envelope: the paper brightness nearby, from far around.
        // (Computed on a copy 8x smaller: brightest value of each 8x8 block first, so nothing is lost.)
        val env = Mat()
        run {
            val shrink = 8
            val k8 = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(shrink.toDouble(), shrink.toDouble()))
            val blockMax = Mat(); Imgproc.dilate(g, blockMax, k8); k8.release()
            val sw = max(1, (w + shrink - 1) / shrink); val sh = max(1, (h + shrink - 1) / shrink)
            val tiny = Mat(); Imgproc.resize(blockMax, tiny, Size(sw.toDouble(), sh.toDouble()), 0.0, 0.0, Imgproc.INTER_NEAREST); blockMax.release()
            val big = odd(max(3, max(w, h) / 4 / shrink)).toDouble()
            val kb = Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(big, big))
            Imgproc.dilate(tiny, tiny, kb); kb.release()
            Imgproc.GaussianBlur(tiny, tiny, Size(0.0, 0.0), max(w, h) / 20.0 / shrink)
            Imgproc.resize(tiny, env, Size(w.toDouble(), h.toDouble()), 0.0, 0.0, Imgproc.INTER_LINEAR); tiny.release()
        }
        val envA = FloatArray(n); env.get(0, 0, envA); env.release()
        val gA = FloatArray(n); g.get(0, 0, gA); g.release()

        val candB = ByteArray(n)
        for (i in 0 until n) if (edgeB[i].toInt() == 0 && gA[i] > 0.30f * envA[i]) candB[i] = 1
        val cand = Mat(h, w, CvType.CV_8UC1); cand.put(0, 0, candB)

        val labels = Mat(); val stats = Mat(); val cents = Mat()
        val count = Imgproc.connectedComponentsWithStats(cand, labels, stats, cents, 4, CvType.CV_32S)
        cand.release(); cents.release()
        val lab = IntArray(n); labels.get(0, 0, lab); labels.release()
        val area = IntArray(count) { if (it == 0) 0 else stats.get(it, Imgproc.CC_STAT_AREA)[0].toInt() }
        stats.release()
        val maxArea = area.maxOrNull() ?: 0
        val accepted = BooleanArray(count) { it != 0 && area[it] >= max(maxArea * 0.25, n * 0.02) }
        if (accepted.any { it }) {
            // Spread outwards: a fenced-in region joins the paper if it looks like the accepted paper
            // around it. "Around it" = a wide blur of the accepted pixels, which is done on a copy 4x
            // smaller (the same result, a fraction of the time), and every 2nd pixel is enough to judge.
            val sf = Mat(); small.convertTo(sf, CvType.CV_32F)
            val sA = FloatArray(n * 3); sf.get(0, 0, sA)
            val minArea = n * 0.0003
            val shrink = 4
            val qw = max(1, (w + shrink - 1) / shrink); val qh = max(1, (h + shrink - 1) / shrink)
            val qSize = Size(qw.toDouble(), qh.toDouble())
            val m8 = Mat(h, w, CvType.CV_8UC1)
            val mB = ByteArray(n)
            val wsum = Mat(); val num = Mat(); val prod = Mat()
            val wA = FloatArray(qw * qh); val nA = FloatArray(qw * qh * 3)
            for (iter in 0 until 4) {
                for (i in 0 until n) mB[i] = if (accepted[lab[i]]) 255.toByte() else 0
                m8.put(0, 0, mB)
                Imgproc.resize(m8, wsum, qSize, 0.0, 0.0, Imgproc.INTER_AREA)
                wsum.convertTo(wsum, CvType.CV_32F, 1.0 / 255)
                Imgproc.GaussianBlur(wsum, wsum, Size(0.0, 0.0), 12.0 / shrink)
                prod.create(h, w, CvType.CV_32FC3); prod.setTo(Scalar(0.0, 0.0, 0.0))
                sf.copyTo(prod, m8)
                Imgproc.resize(prod, num, qSize, 0.0, 0.0, Imgproc.INTER_AREA)
                Imgproc.GaussianBlur(num, num, Size(0.0, 0.0), 12.0 / shrink)
                wsum.get(0, 0, wA); num.get(0, 0, nA)
                val sums = DoubleArray(count); val cnt = IntArray(count)
                var y = 0
                while (y < h) {
                    val qy = min(qh - 1, y / shrink) * qw
                    var x = 0
                    while (x < w) {
                        val i = y * w + x
                        val l = lab[i]
                        if (l != 0 && !accepted[l]) {
                            val qi = qy + min(qw - 1, x / shrink)
                            val ws = wA[qi]
                            if (ws >= 0.05f) {
                                var dev = 0f
                                for (c in 0 until 3) {
                                    val local = nA[qi * 3 + c] / ws
                                    val d = abs(sA[i * 3 + c] / max(local, 1f) - 1f)
                                    if (d > dev) dev = d
                                }
                                sums[l] += dev.toDouble(); cnt[l]++
                            }
                        }
                        x += 2
                    }
                    y += 2
                }
                var changed = false
                for (l in 1 until count) {
                    if (accepted[l] || area[l] < minArea || cnt[l] < min(area[l] / 4, 8)) continue
                    if (sums[l] / cnt[l] < 0.10) { accepted[l] = true; changed = true }
                }
                if (!changed) break
            }
            listOf(sf, m8, wsum, num, prod).forEach { it.release() }
        } else {
            for (l in 1 until count) accepted[l] = true // nothing big found: use every candidate
        }
        val outA = FloatArray(n)
        for (i in 0 until n) if (candB[i].toInt() == 1 && accepted[lab[i]]) outA[i] = 1f
        val out = Mat(h, w, CvType.CV_32FC1); out.put(0, 0, outA)
        return out
    }

    /** Fills the holes in [value] (where [weight] is 0) smoothly from the surrounding paper. */
    private fun pushPull(value: Mat, weight: Mat): Mat {
        val vs = mutableListOf<Mat>(); val ws = mutableListOf<Mat>()
        val w3 = Mat(); Core.merge(listOf(weight, weight, weight), w3)
        val v0 = Mat(); Core.multiply(value, w3, v0); w3.release()
        vs += v0; ws += weight.clone()
        while (min(vs.last().cols(), vs.last().rows()) > 8) {
            val v = Mat(); val ww = Mat()
            Imgproc.pyrDown(vs.last(), v); Imgproc.pyrDown(ws.last(), ww)
            vs += v; ws += ww
        }
        fun ratio(v: Mat, ww: Mat): Mat {
            val d = Mat(); Core.max(ww, Scalar(1e-6), d)
            val d3 = Mat(); Core.merge(listOf(d, d, d), d3); d.release()
            val r = Mat(); Core.divide(v, d3, r); d3.release()
            return r
        }
        var est = ratio(vs.last(), ws.last())
        for (l in vs.size - 2 downTo 0) {
            val up = Mat()
            Imgproc.pyrUp(est, up, Size(vs[l].cols().toDouble(), vs[l].rows().toDouble()))
            est.release()
            val r = ratio(vs[l], ws[l])
            val a = Mat(); Core.multiply(ws[l], Scalar(4.0), a); Core.min(a, Scalar(1.0), a)
            val a3 = Mat(); Core.merge(listOf(a, a, a), a3); a.release()
            val diff = Mat(); Core.subtract(r, up, diff); r.release()
            Core.multiply(diff, a3, diff); a3.release()
            val next = Mat(); Core.add(up, diff, next)
            diff.release(); up.release()
            est = next
        }
        vs.forEach { it.release() }; ws.forEach { it.release() }
        return est
    }

    /** Tone curve on lightness only (colours stay true), sharpen lightness, adjust colourfulness. */
    private fun toneL(flat: Mat, white: Double, blackPct: Double, gamma: Double, sharpenAmount: Double, chroma: Double): Mat {
        val lab = Mat()
        Imgproc.cvtColor(flat, lab, Imgproc.COLOR_BGR2Lab)
        val parts = ArrayList<Mat>(3)
        Core.split(lab, parts)
        val black = min(120.0, percentile(parts[0], blackPct))
        val lv = levels(parts[0], black, 255 * white, gamma)
        parts[0].release()
        parts[0] = if (sharpenAmount > 0) sharpen(lv, sharpenAmount).also { lv.release() } else lv
        if (chroma != 1.0) {
            parts[1].convertTo(parts[1], -1, chroma, 128 * (1 - chroma))
            parts[2].convertTo(parts[2], -1, chroma, 128 * (1 - chroma))
        }
        Core.merge(parts, lab)
        parts.forEach { it.release() }
        val out = Mat()
        Imgproc.cvtColor(lab, out, Imgproc.COLOR_Lab2BGR)
        lab.release()
        return out
    }

    /** For photos and colourful pages: brighter, balanced colours, no hard cleaning. */
    private fun lighten(bgr: Mat): Mat {
        val lab = Mat()
        Imgproc.cvtColor(bgr, lab, Imgproc.COLOR_BGR2Lab)
        val parts = ArrayList<Mat>(3)
        Core.split(lab, parts)
        val clahe = Imgproc.createCLAHE(2.0, Size(8.0, 8.0))
        val l2 = Mat()
        clahe.apply(parts[0], l2)
        parts[0].release()
        parts[0] = l2
        Core.merge(parts, lab)
        parts.forEach { it.release() }
        val out = Mat()
        Imgproc.cvtColor(lab, out, Imgproc.COLOR_Lab2BGR)
        lab.release()
        val g = Mat()
        Imgproc.cvtColor(out, g, Imgproc.COLOR_BGR2GRAY)
        val thr = percentile(g, 95.0)
        val mask = Mat()
        Imgproc.threshold(g, mask, thr - 1, 255.0, Imgproc.THRESH_BINARY)
        val mean = Core.mean(out, mask)
        g.release(); mask.release()
        val gains = Scalar(
            min(1.6, 230.0 / max(1.0, mean.`val`[0])),
            min(1.6, 230.0 / max(1.0, mean.`val`[1])),
            min(1.6, 230.0 / max(1.0, mean.`val`[2])),
        )
        Core.multiply(out, gains, out)
        return sharpen(out, 0.4, 1.2).also { out.release() }
    }

    /** Maps [black]..[white] to 0..255 with a gamma curve (works on 1 or 3 channels). */
    private fun levels(img: Mat, black: Double, white: Double, gamma: Double): Mat {
        val table = ByteArray(256) { v ->
            val t = ((v - black) / max(1.0, white - black)).coerceIn(0.0, 1.0)
            (Math.pow(t, gamma) * 255).roundToInt().coerceIn(0, 255).toByte()
        }
        val lut = Mat(1, 256, CvType.CV_8UC1)
        lut.put(0, 0, table)
        val out = Mat()
        Core.LUT(img, lut, out)
        lut.release()
        return out
    }

    private fun blackPoint(gray: Mat): Double = percentile(gray, 0.5).coerceIn(0.0, 120.0)

    private fun percentile(gray: Mat, pct: Double): Double {
        val hist = Mat()
        Imgproc.calcHist(listOf(gray), MatOfInt(0), Mat(), hist, MatOfInt(256), MatOfFloat(0f, 256f))
        val counts = FloatArray(256)
        hist.get(0, 0, counts)
        hist.release()
        val total = counts.sum()
        var acc = 0f
        for (i in 0 until 256) {
            acc += counts[i]
            if (acc >= total * pct / 100) return i.toDouble()
        }
        return 255.0
    }

    private fun sharpen(img: Mat, amount: Double, sigma: Double = 1.0): Mat {
        val blur = Mat()
        Imgproc.GaussianBlur(img, blur, Size(0.0, 0.0), sigma)
        val out = Mat()
        Core.addWeighted(img, 1 + amount, blur, -amount, 0.0, out)
        blur.release()
        return out
    }

    private fun odd(n: Int) = if (n % 2 == 0) n + 1 else n

    // ---------- Full page render ----------

    /** Straighten + filter + rotate + drawings. Used for the final page (big) and sharp zoom images. */
    fun render(file: File, quad: Quad, filter: ScanFilter, rotation: Int, maxSide: Int, marks: List<Mark> = emptyList()): Mat {
        val page = warpPage(file, quad, maxSide)
        val out = applyFilter(page, filter)
        page.release()
        rotateInPlace(out, rotation)
        return withMarks(out, marks)
    }

    // ---------- Drawings on the page (pen, highlighter, signature, typed text) ----------

    private fun bgr(argb: Int) = Scalar((argb and 0xFF).toDouble(), ((argb shr 8) and 0xFF).toDouble(), ((argb shr 16) and 0xFF).toDouble())

    /**
     * Draws the marks onto a finished page. Returns the image to use: [page] itself, or a colour
     * copy when [page] was grey (then [page] has been released).
     */
    fun withMarks(page: Mat, marks: List<Mark>): Mat {
        if (marks.isEmpty()) return page
        var img = page
        if (img.channels() == 1) {
            img = Mat()
            Imgproc.cvtColor(page, img, Imgproc.COLOR_GRAY2BGR)
            page.release()
        }
        val w = img.cols(); val h = img.rows()
        val unit = min(w, h)
        for (m in marks) {
            when (m) {
                is Mark.Stroke -> {
                    val thick = max(1, (m.width * unit).roundToInt()).coerceAtMost(255)
                    val pts = m.points.map { Point(it.x * w.toDouble(), it.y * h.toDouble()) }
                    if (pts.isEmpty()) continue
                    if (!m.highlighter) {
                        drawStroke(img, pts, bgr(m.color), thick)
                    } else {
                        // A highlighter only tints: multiply, so the text under it stays fully dark.
                        val pad = thick / 2 + 2
                        val x0 = max(0, pts.minOf { it.x }.toInt() - pad); val y0 = max(0, pts.minOf { it.y }.toInt() - pad)
                        val x1 = min(w, pts.maxOf { it.x }.toInt() + pad + 1); val y1 = min(h, pts.maxOf { it.y }.toInt() + pad + 1)
                        if (x1 <= x0 || y1 <= y0) continue
                        val roi = img.submat(y0, y1, x0, x1)
                        val ink = Mat(roi.rows(), roi.cols(), CvType.CV_8UC3, Scalar(255.0, 255.0, 255.0))
                        drawStroke(ink, pts.map { Point(it.x - x0, it.y - y0) }, bgr(m.color), thick)
                        Core.multiply(roi, ink, roi, 1.0 / 255)
                        ink.release(); roi.release()
                    }
                }
                is Mark.Label -> runCatching { drawLabel(img, m) }
            }
        }
        return img
    }

    private fun drawStroke(img: Mat, pts: List<Point>, color: Scalar, thick: Int) {
        if (pts.size == 1) {
            Imgproc.circle(img, pts[0], max(1, thick / 2), color, -1, Imgproc.LINE_AA)
        } else {
            val mp = org.opencv.core.MatOfPoint(*pts.toTypedArray())
            Imgproc.polylines(img, listOf(mp), false, color, thick, Imgproc.LINE_AA)
            mp.release()
        }
    }

    /** Typed text: drawn with the phone's own fonts (so any language works), then blended onto the page. */
    private fun drawLabel(img: Mat, m: Mark.Label) {
        val w = img.cols(); val h = img.rows()
        val px = max(8f, m.size * min(w, h))
        val paint = android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG).apply {
            color = m.color or 0xFF000000.toInt()
            textSize = px
        }
        val lines = m.text.split("\n")
        val fm = paint.fontMetrics
        val lineH = fm.descent - fm.ascent
        val tw = (lines.maxOf { paint.measureText(it) }.toInt() + 5).coerceAtLeast(1)
        val th = ((lineH * lines.size).toInt() + 5).coerceAtLeast(1)
        if (tw.toLong() * th > 12_000_000L) return
        val bmp = Bitmap.createBitmap(tw, th, Bitmap.Config.ARGB_8888)
        val canvas = android.graphics.Canvas(bmp)
        lines.forEachIndexed { i, line -> canvas.drawText(line, 2f, 2f - fm.ascent + i * lineH, paint) }
        val pixels = IntArray(tw * th)
        bmp.getPixels(pixels, 0, tw, 0, 0, tw, th)
        bmp.recycle()
        // Only the coverage (alpha) of each pixel is needed: the colour is the ink colour.
        val alpha = ByteArray(tw * th) { (pixels[it] ushr 24).toByte() }
        val label = Mat(th, tw, CvType.CV_8UC1)
        label.put(0, 0, alpha)
        rotateInPlace(label, m.angle)
        val lw = label.cols(); val lh = label.rows()
        val left = (m.x * w - lw / 2f).roundToInt(); val top = (m.y * h - lh / 2f).roundToInt()
        val x0 = max(0, left); val y0 = max(0, top); val x1 = min(w, left + lw); val y1 = min(h, top + lh)
        if (x1 > x0 && y1 > y0) {
            val cover = label.submat(y0 - top, y1 - top, x0 - left, x1 - left)
            val dst = img.submat(y0, y1, x0, x1)
            val cw = x1 - x0; val ch = y1 - y0
            val a = ByteArray(cw * ch); cover.get(0, 0, a)
            val d = ByteArray(cw * ch * 3); dst.get(0, 0, d)
            val ink = intArrayOf(m.color and 0xFF, (m.color shr 8) and 0xFF, (m.color shr 16) and 0xFF) // B, G, R
            for (i in 0 until cw * ch) {
                val k = a[i].toInt() and 0xFF
                if (k == 0) continue
                for (c in 0 until 3) {
                    val old = d[i * 3 + c].toInt() and 0xFF
                    d[i * 3 + c] = ((old * (255 - k) + ink[c] * k) / 255).toByte()
                }
            }
            dst.put(0, 0, d)
            cover.release(); dst.release()
        }
        label.release()
    }

    /**
     * The straightened page (no filter), at most [maxSide] px. [fast] is for on-screen previews:
     * it accepts a slightly smaller decode and uses quicker resampling.
     */
    fun warpPage(file: File, quad: Quad, maxSide: Int, fast: Boolean = false): Mat {
        val target = if (maxSide > 0) maxSide else MAX_SIDE
        // Load the photo big enough that the PAGE (not the whole photo) gets [target] pixels:
        // a page filling 60% of the photo needs the photo at target / 0.6. Fixes soft previews.
        val (w, h) = imageSize(file).let { (a, b) -> if (exifDegrees(file) % 180 == 90) b to a else a to b }
        val longest = max(w, h).coerceAtLeast(1)
        val pts = quad.points
        val pageW = max(hypot((pts[1].x - pts[0].x) * w.toDouble(), (pts[1].y - pts[0].y) * h.toDouble()),
            hypot((pts[2].x - pts[3].x) * w.toDouble(), (pts[2].y - pts[3].y) * h.toDouble()))
        val pageH = max(hypot((pts[3].x - pts[0].x) * w.toDouble(), (pts[3].y - pts[0].y) * h.toDouble()),
            hypot((pts[2].x - pts[1].x) * w.toDouble(), (pts[2].y - pts[1].y) * h.toDouble()))
        val fraction = (max(pageW, pageH) / longest).coerceIn(0.15, 1.0)
        val src = load(file, min(longest, (target * 1.1 / fraction).roundToInt()), soft = fast)
        val flat = warp(src, quad, if (fast) Imgproc.INTER_LINEAR else Imgproc.INTER_CUBIC)
        src.release()
        resizeToFit(flat, target)
        return flat
    }

    /**
     * How sharp the page in a photo is: about 0.72-0.87 for normal sharp photos, below 0.6 for
     * clearly blurry or shaken ones. Measures how much detail a further small blur would remove
     * (a blurry photo has little left to lose), on the middle of the page at a fixed size.
     */
    fun sharpness(file: File, quad: Quad): Double {
        val page = warpPage(file, quad, 1600)
        val g = Mat()
        Imgproc.cvtColor(page, g, Imgproc.COLOR_BGR2GRAY)
        page.release()
        val r = g.submat(g.rows() / 6, g.rows() * 5 / 6, g.cols() / 6, g.cols() * 5 / 6)
        val f = Mat(); r.convertTo(f, CvType.CV_32F); r.release(); g.release()
        val w = f.cols(); val h = f.rows()
        val a = FloatArray(w * h); f.get(0, 0, a)
        val bv = Mat(); val bh = Mat()
        Imgproc.blur(f, bv, Size(1.0, 9.0)); Imgproc.blur(f, bh, Size(9.0, 1.0))
        f.release()
        val av = FloatArray(w * h); bv.get(0, 0, av); bv.release()
        val ah = FloatArray(w * h); bh.get(0, 0, ah); bh.release()
        var sv = 0.0; var lv = 0.0; var sh = 0.0; var lh = 0.0
        for (y in 0 until h) for (x in 0 until w) {
            val i = y * w + x
            if (y + 1 < h) {
                val d = abs(a[i + w] - a[i]); val db = abs(av[i + w] - av[i])
                sv += d; lv += max(0f, d - db)
            }
            if (x + 1 < w) {
                val d = abs(a[i + 1] - a[i]); val db = abs(ah[i + 1] - ah[i])
                sh += d; lh += max(0f, d - db)
            }
        }
        if (sv < 1 || sh < 1) return 1.0 // blank page: nothing to judge
        return min(lv / sv, lh / sh)
    }

    /** Below this, the photo is clearly blurry (sharp photos measured 0.72 or more). */
    const val BLURRY = 0.6

    /** Saves a page as JPEG. Grey and B&W pages are saved as single-channel JPEGs (smaller files). */
    fun saveJpeg(mat: Mat, file: File, quality: Int = 93) {
        // Always a normal colour JPEG: every PDF tool and viewer handles it.
        val out = if (mat.channels() == 1) Mat().also { Imgproc.cvtColor(mat, it, Imgproc.COLOR_GRAY2BGR) } else mat
        val params = MatOfInt(Imgcodecs.IMWRITE_JPEG_QUALITY, quality)
        val ok = Imgcodecs.imwrite(file.path, out, params)
        params.release()
        if (out !== mat) out.release()
        if (!ok) error("Could not save the page")
    }

    /** Rotates a saved page file in place by 90° steps, in native memory (no big Java bitmaps). */
    fun rotateFile(src: File, degrees: Int, dest: File) {
        val mat = Imgcodecs.imread(src.path, Imgcodecs.IMREAD_COLOR or Imgcodecs.IMREAD_IGNORE_ORIENTATION)
        if (mat.empty()) { mat.release(); error("Could not read the page") }
        rotateInPlace(mat, degrees)
        saveJpeg(mat, dest)
        mat.release()
    }

    /**
     * ID card sheet: front and back placed at real size (85.6 x 54 mm) on a white A4 page at 300 dpi,
     * ready to print or attach to forms.
     */
    fun composeIdCard(front: Mat, back: Mat, dest: File) {
        val pageW = 2480; val pageH = 3508          // A4 at 300 dpi
        val cardW = 1011; val cardH = 638           // ID-1 card at 300 dpi
        val sheet = Mat(pageH, pageW, CvType.CV_8UC3, Scalar(255.0, 255.0, 255.0))
        listOf(front to (pageH * 0.18).toInt(), back to (pageH * 0.55).toInt()).forEach { (card, top) ->
            val c = if (card.channels() == 1) Mat().also { Imgproc.cvtColor(card, it, Imgproc.COLOR_GRAY2BGR) } else card.clone()
            if (c.rows() > c.cols()) Core.rotate(c, c, Core.ROTATE_90_COUNTERCLOCKWISE) // cards lie landscape
            val s = min(cardW.toDouble() / c.cols(), cardH.toDouble() / c.rows())
            val sized = Mat()
            Imgproc.resize(c, sized, Size(c.cols() * s, c.rows() * s), 0.0, 0.0, if (s < 1) Imgproc.INTER_AREA else Imgproc.INTER_CUBIC)
            c.release()
            val left = (pageW - sized.cols()) / 2
            val roi = sheet.submat(top, top + sized.rows(), left, left + sized.cols())
            sized.copyTo(roi)
            roi.release()
            Imgproc.rectangle(sheet, Point(left - 2.0, top - 2.0), Point(left + sized.cols() + 1.0, top + sized.rows() + 1.0), Scalar(200.0, 200.0, 200.0), 2)
            sized.release()
        }
        saveJpeg(sheet, dest)
        sheet.release()
    }

    /** Like [toBitmap] but half the memory (RGB_565): for big zoom images. */
    fun toBitmapSmall(mat: Mat): Bitmap {
        val rgb = Mat()
        when (mat.channels()) {
            1 -> Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_GRAY2RGB)
            else -> Imgproc.cvtColor(mat, rgb, Imgproc.COLOR_BGR2RGB)
        }
        return try {
            Bitmap.createBitmap(rgb.cols(), rgb.rows(), Bitmap.Config.RGB_565).also { Utils.matToBitmap(rgb, it) }
        } catch (e: UnsatisfiedLinkError) {
            toBitmap(mat) // PC tests only (see copyInto)
        } finally {
            rgb.release()
        }
    }

    /** Converts any 1- or 3-channel image into a Bitmap for showing on screen. */
    fun toBitmap(mat: Mat): Bitmap {
        val rgba = Mat()
        when (mat.channels()) {
            1 -> Imgproc.cvtColor(mat, rgba, Imgproc.COLOR_GRAY2RGBA)
            3 -> Imgproc.cvtColor(mat, rgba, Imgproc.COLOR_BGR2RGBA)
            else -> mat.copyTo(rgba)
        }
        val bmp = Bitmap.createBitmap(rgba.cols(), rgba.rows(), Bitmap.Config.ARGB_8888)
        copyInto(rgba, bmp)
        rgba.release()
        return bmp
    }

    /**
     * Copies an RGBA image into [bmp]. On a phone OpenCV writes straight into the bitmap. The PC
     * tests use a desktop OpenCV without that Android part, so there the bytes are copied instead.
     */
    private fun copyInto(rgba: Mat, bmp: Bitmap) {
        try {
            Utils.matToBitmap(rgba, bmp)
        } catch (e: UnsatisfiedLinkError) {
            val bytes = ByteArray(rgba.cols() * rgba.rows() * 4)
            rgba.get(0, 0, bytes)
            bmp.copyPixelsFromBuffer(java.nio.ByteBuffer.wrap(bytes))
        }
    }

    /** Makes a Mat from a camera/gallery Bitmap (for gallery images we first save them as JPEG). */
    @Suppress("unused")
    fun fromBitmap(bitmap: Bitmap): Mat {
        val rgba = Mat(bitmap.height, bitmap.width, CvType.CV_8UC4)
        Utils.bitmapToMat(bitmap, rgba)
        val bgr = Mat()
        Imgproc.cvtColor(rgba, bgr, Imgproc.COLOR_RGBA2BGR)
        rgba.release()
        return bgr
    }
}
