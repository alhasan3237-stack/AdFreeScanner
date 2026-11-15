package com.myscanner.app.scan

import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.MatOfInt
import org.opencv.core.MatOfPoint
import org.opencv.core.MatOfPoint2f
import org.opencv.core.Point
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.acos
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt
import kotlin.math.sign
import kotlin.math.sin
import kotlin.math.sqrt

/**
 * Finds the page in a photo (version 3: 91 of 100 hard test photos exact, vs 49 of 100 before).
 *
 * 1. Text is erased (morphological closing) so only big shapes remain.
 * 2. Possible outlines come from two sources: straight edge lines (combined in pairs) and outlines of blobs.
 * 3. Each outline is scored:
 *    - every side must sit on a real edge running in the same direction,
 *    - every side must have paper on the inside and table on the outside, consistently along its
 *      whole length (rejects pattern lines, pens and shadow lines),
 *    - corners must be real corners: the edge must not carry on past them (rejects shapes cut off by
 *      a shadow),
 *    - sensible corner angles and size.
 * 4. The winner's sides are refined by fitting straight lines to the edge pixels.
 * 5. The outline is pulled slightly inside the page ([INSET]) so no table shows at the borders.
 *
 * Works on colour (better) or greyscale (live camera) images, ideally 400–640 px on the long side.
 */
object DocumentDetector {

    /** How far inside the page edge the outline sits, as a fraction of the page's shorter side. */
    const val INSET = 0.012

    private const val GRAD_T = 15f
    private const val NET_SURE = 0.6

    /** Details of the last decision (for the PC tests only). */
    @Volatile var lastDebug: String = ""
    /** Tests only: when true, [lastTrace] gets the candidate counts and the best outlines with their scores. */
    @Volatile var trace: Boolean = false
    @Volatile var lastTrace: String = ""
    @Volatile private var lineTrace: String = ""
    /** Tests only: [edge-based outline, network-based outline, numbers] of the last photo. */
    @Volatile var lastOptions: List<String> = emptyList()
    /** Tests only: folder to write intermediate images into. */
    @Volatile var traceDir: String? = null


    /** Corner network: score multiplier for its outline among the candidates. */
    private const val DQ_WEIGHT = 1.0
    /**
     * When the edges are weak (best edge score below this) the corner network decides WHICH page,
     * and the edges only place its corners precisely. Tuned on the app's own results for 196 test
     * images (35 hand-labelled real photos), each in 6 slightly different versions (noise, JPEG, blur...).
     */
    private const val WEAK_EDGES = 0.065
    /** ...also when the small page network disagrees with the edge outline and the corner network is fairly sure. */
    private const val NET_DISAGREE = 0.4
    private const val DQ_FAIRLY_SURE = 0.3
    /** Overlap needed to count an edge outline as "the same page" as the corner network's. */
    private const val SAME_PAGE = 0.75
    /** ...and when the two clearly disagree (overlap below this) while the corner network is sure enough. */
    private const val DISAGREE = 0.7
    private const val DQ_SURE = 0.7
    /** Below this confidence the corner network has no say at all. */
    private const val DQ_MIN = 0.25

    /** Overlap between an outline and the network's page area (intersection over union, 0..1). */
    private class NetOverlap(private val net: Mat, private val w: Int, private val h: Int) {
        private val poly = Mat.zeros(net.rows(), net.cols(), CvType.CV_8UC1)
        private val tmp = Mat()
        private val netCount = Core.countNonZero(net)

        fun iou(q: List<V>): Double {
            poly.setTo(org.opencv.core.Scalar(0.0))
            val sx = net.cols().toDouble() / w; val sy = net.rows().toDouble() / h
            val pts = MatOfPoint(*q.map { Point(it.x * sx, it.y * sy) }.toTypedArray())
            Imgproc.fillConvexPoly(poly, pts, org.opencv.core.Scalar(255.0))
            pts.release()
            Core.bitwise_and(poly, net, tmp)
            val inter = Core.countNonZero(tmp)
            val union = Core.countNonZero(poly) + netCount - inter
            return if (union > 0) inter.toDouble() / union else 0.0
        }

        fun release() { poly.release(); tmp.release() }
    }
    private const val MAX_LINES = 10

    private class V(val x: Double, val y: Double) {
        operator fun plus(o: V) = V(x + o.x, y + o.y)
        operator fun minus(o: V) = V(x - o.x, y - o.y)
        operator fun times(s: Double) = V(x * s, y * s)
        fun len() = hypot(x, y)
    }

    private class Seg(val a: V, val b: V, val length: Double, val theta: Double)

    /** Per-image data shared by all scoring. */
    private class Img(
        val w: Int,
        val h: Int,
        val chans: List<ByteArray>, // text-free channels: [lightness, colourfulness?]
        val gx: FloatArray,
        val gy: FloatArray,
        val mag: FloatArray,
        /** 1 where there is printing (dark detail the text-erasing step removed). */
        val text: ByteArray,
    ) {
        fun v(ch: Int, x: Int, y: Int): Int = chans[ch][y * w + x].toInt() and 0xFF
        val maxDim get() = max(w, h)
    }

    /** A detected page outline, and whether the detector is sure about it. */
    class Result(val quad: Quad, val sure: Boolean)

    /** For the live camera outline: same checks as for photos; [cornerHint] = a recent corner-network result to reuse. */
    fun detect(src: Mat, cornerHint: DocQuad.Result? = null): Quad? = detectWithConfidence(src, useNet = true, cornerHint = cornerHint)?.quad

    /**
     * Like [detect], plus a confidence check: not sure when the best outline is weak, or when a
     * clearly different outline scores almost as well (e.g. a page lying on another white sheet).
     * Sure = the trained network agrees with the outline (overlap at least 0.6).
     */
    fun detectWithConfidence(src: Mat, useNet: Boolean = true, cornerHint: DocQuad.Result? = null): Result? {
        val w = src.cols()
        val h = src.rows()
        if (w < 48 || h < 48) return null
        val toRelease = mutableListOf<Mat>()
        fun <T : Mat> keep(m: T): T { toRelease += m; return m }
        try {
            // ---- channels: lightness (+ colourfulness for colour photos) ----
            val raw = mutableListOf<Mat>()
            if (src.channels() >= 3) {
                val lab = keep(Mat())
                Imgproc.cvtColor(src, lab, Imgproc.COLOR_BGR2Lab)
                val parts = ArrayList<Mat>(3)
                Core.split(lab, parts)
                parts.forEach { keep(it) }
                val a = keep(Mat()); val b = keep(Mat()); val m = keep(Mat()); val chroma = keep(Mat())
                parts[1].convertTo(a, CvType.CV_32F, 1.0, -128.0)
                parts[2].convertTo(b, CvType.CV_32F, 1.0, -128.0)
                Core.magnitude(a, b, m)
                m.convertTo(chroma, CvType.CV_8U, 3.0, 0.0) // boost colour differences
                raw += parts[0]; raw += chroma
            } else {
                raw += src
            }

            // ---- erase text ----
            val k = (max(5, max(w, h) / 60)) or 1
            val kernel = keep(Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(k.toDouble(), k.toDouble())))
            val smooth = raw.map { c ->
                val blur = keep(Mat()); val closed = keep(Mat())
                Imgproc.GaussianBlur(c, blur, Size(5.0, 5.0), 0.0)
                Imgproc.morphologyEx(blur, closed, Imgproc.MORPH_CLOSE, kernel)
                closed
            }

            // ---- gradients (strongest channel wins) ----
            val n = w * h
            val gx = FloatArray(n); val gy = FloatArray(n); val mag = FloatArray(n)
            val tx = FloatArray(n); val ty = FloatArray(n)
            for (c in smooth) {
                val dx = keep(Mat()); val dy = keep(Mat())
                Imgproc.Sobel(c, dx, CvType.CV_32F, 1, 0, 3)
                Imgproc.Sobel(c, dy, CvType.CV_32F, 0, 1, 3)
                dx.get(0, 0, tx); dy.get(0, 0, ty)
                for (i in 0 until n) {
                    val m = sqrt(tx[i] * tx[i] + ty[i] * ty[i])
                    if (m > mag[i]) { mag[i] = m; gx[i] = tx[i]; gy[i] = ty[i] }
                }
            }
            val chanBytes = smooth.map { c -> ByteArray(n).also { c.get(0, 0, it) } }
            // Printing: places much darker than the text-free version.
            val rawL = ByteArray(n).also { raw[0].get(0, 0, it) }
            val textMat = keep(Mat(h, w, CvType.CV_8UC1))
            val tb = ByteArray(n)
            for (i in 0 until n) if ((chanBytes[0][i].toInt() and 0xFF) - (rawL[i].toInt() and 0xFF) > 25) tb[i] = 1
            textMat.put(0, 0, tb)
            Imgproc.dilate(textMat, textMat, keep(Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0))))
            textMat.get(0, 0, tb)
            val img = Img(w, h, chanBytes, gx, gy, mag, tb)

            // ---- edges ----
            val edges = keep(Mat.zeros(h, w, CvType.CV_8UC1))
            for (c in smooth) {
                val e = keep(Mat())
                Imgproc.Canny(c, e, 10.0, 30.0) // text is gone, so low thresholds are safe
                Core.bitwise_or(edges, e, edges)
            }
            val edgeBytes = ByteArray(n).also { edges.get(0, 0, it) }
            traceDir?.let { dir ->
                org.opencv.imgcodecs.Imgcodecs.imwrite("$dir/k_edges.png", edges)
                smooth.forEachIndexed { i, c -> org.opencv.imgcodecs.Imgcodecs.imwrite("$dir/k_smooth$i.png", c) }
                raw.forEachIndexed { i, c -> org.opencv.imgcodecs.Imgcodecs.imwrite("$dir/k_raw$i.png", c) }
            }

            // ---- the corner network (the live camera passes a recent result instead of running it every frame) ----
            val dq: DocQuad.Result? = if (useNet) cornerHint ?: DocQuad.run(src) else null

            // ---- the small page network's view of where the page is ----
            val netMask: Mat? = if (useNet) PageNet.predict(src)?.let { prob ->
                val m8 = keep(Mat())
                Imgproc.threshold(prob, prob, 0.5, 255.0, Imgproc.THRESH_BINARY)
                prob.convertTo(m8, CvType.CV_8U)
                prob.release()
                m8
            } else null

            // ---- candidate outlines ----
            val candidates = mutableListOf<List<V>>()
            // (Tested: the network works best as a check on outlines, not as a source of them.)
            val overlap = netMask?.let { NetOverlap(it, w, h) }
            val dil = keep(Mat())
            val k3 = keep(Imgproc.getStructuringElement(Imgproc.MORPH_RECT, Size(3.0, 3.0)))
            Imgproc.dilate(edges, dil, k3)
            val otsu = keep(Mat())
            Imgproc.threshold(smooth[0], otsu, 0.0, 255.0, Imgproc.THRESH_BINARY + Imgproc.THRESH_OTSU)
            candidates += contourQuads(dil, w * h.toDouble())
            candidates += contourQuads(otsu, w * h.toDouble())
            candidates += lineQuads(edges, edgeBytes, img)

            val dqQuad: List<V>? = dq?.corners?.map {
                V((it.first * w).coerceIn(0.0, w - 1.0), (it.second * h).coerceIn(0.0, h - 1.0))
            }
            val nBase = candidates.size
            if (dqQuad != null) candidates += dqQuad

            // ---- score ----
            val scored = mutableListOf<Pair<Double, List<V>>>()
            for ((ci, c) in candidates.withIndex()) {
                val q = order(c) ?: continue
                var s = score(q, img)
                if (ci >= nBase) s *= DQ_WEIGHT
                if (s > 0 && overlap != null) {
                    val iou = overlap.iou(q)
                    s *= Math.pow(0.15 + 0.85 * iou * iou * iou, 0.3)
                }
                if (s > 0) scored += s to q
            }
            if (scored.isEmpty() && dqQuad == null) return null
            scored.sortByDescending { it.first }
            if (trace) {
                val sb = StringBuilder("candidates=${candidates.size} valid=${scored.size}\n")
                scored.take(4).forEach { (sc, q) ->
                    sb.append("  %.4f  ".format(sc)).append(q.joinToString(" ") { "(%.0f,%.0f)".format(it.x, it.y) })
                        .append("  ").append(scoreParts(q, img)).append("\n")
                }
                lastTrace = sb.toString() + lineTrace
            }
            val dqOrdered = dqQuad?.let { order(it) }
            var best: Double
            var winner: List<V>
            if (scored.isNotEmpty()) {
                best = scored[0].first; winner = scored[0].second
            } else {
                winner = dqOrdered ?: return null; best = 0.0
            }
            // Weak edges, or the page network disagrees: the corner network picks the page.
            var switched = false
            var preSwitch = ""
            if (dq != null && dqOrdered != null) {
                val netV3 = overlap?.iou(winner)
                val firstIou = quadIou(winner, dqOrdered, w, h)
                preSwitch = "s_v3=%.3f iou=%.3f net_v3=%.3f".format(best, firstIou, netV3 ?: -1.0)
                val disagree = firstIou < DISAGREE && dq.confidence >= DQ_SURE
                if (dq.confidence >= DQ_MIN &&
                    (best < WEAK_EDGES || disagree || (netV3 != null && netV3 < NET_DISAGREE && dq.confidence >= DQ_FAIRLY_SURE))
                ) {
                    val same = scored.firstOrNull { (_, q) -> quadIou(q, dqOrdered, w, h) >= SAME_PAGE }
                    winner = same?.second ?: dqOrdered
                    best = max(same?.first ?: 0.0, 0.08)
                    switched = true
                }
            }
            if (trace) {
                // Both options with the numbers the choice is based on (for tuning the choice on test photos).
                fun finish(q: List<V>): String = inset(refine(q, img, edgeBytes)).joinToString(" ") { "%.5f %.5f".format(it.x / w, it.y / h) }
                val classic = scored.firstOrNull()
                val netSide = dqOrdered?.let { d -> scored.firstOrNull { (_, q) -> quadIou(q, d, w, h) >= SAME_PAGE } }
                lastOptions = listOf(
                    classic?.let { finish(it.second) } ?: "",
                    dqOrdered?.let { d -> finish(netSide?.second ?: d) } ?: "",
                    "%.4f %.4f %.4f %.4f %.4f %.4f".format(
                        dq?.confidence ?: -1.0, classic?.first ?: 0.0,
                        if (classic != null && dqOrdered != null) quadIou(classic.second, dqOrdered, w, h) else -1.0,
                        classic?.let { overlap?.iou(it.second) } ?: -1.0,
                        netSide?.first ?: 0.0,
                        dqOrdered?.let { overlap?.iou(it) } ?: -1.0),
                )
            }
            val agreeWithDq = if (dqOrdered != null) quadIou(winner, dqOrdered, w, h) else 1.0
            lastDebug = "dqConf=%.3f best=%.3f switched=%s agree=%.3f candidates=%d".format(dq?.confidence ?: -1.0, best, switched, agreeWithDq, scored.size) + " " + preSwitch
            val diag = hypot(w.toDouble(), h.toDouble())
            val rival = if (scored.isEmpty()) 0.0 else scored.drop(1).firstOrNull { (_, q) ->
                (0 until 4).maxOf { (q[it] - winner[it]).len() } > 0.05 * diag
            }?.first ?: 0.0
            val bestIou = overlap?.iou(winner)
            overlap?.release()
            // Tested on 21 real photos + 136 test scenes: the network's agreement is the reliable
            // signal. (The old "close rival" test flagged most correct real photos.)
            var sure = if (bestIou != null) best >= 0.08 && bestIou >= NET_SURE
                       else best >= 0.2 && rival / best <= 0.6
            if (dq != null) {
                // The two detectors agree on the page: sure. They disagree and the network is sure: check.
                if (agreeWithDq >= 0.85 && dq.confidence >= 0.5) sure = true
                if (agreeWithDq < 0.6 && dq.confidence >= 0.5) sure = false
                if (switched) sure = dq.confidence >= 0.5 && agreeWithDq >= 0.75
            }
            val refined = refine(winner, img, edgeBytes)
            val pulledIn = inset(refined)
            val quad = Quad.ordered(pulledIn.map { P((it.x / w).toFloat().coerceIn(0f, 1f), (it.y / h).toFloat().coerceIn(0f, 1f)) })
            return if (quad.isConvex()) Result(quad, sure) else null
        } finally {
            toRelease.forEach { it.release() }
        }
    }

    /** Overlap (intersection over union) of two outlines in pixels of a w x h picture. */
    private fun quadIou(a: List<V>, b: List<V>, w: Int, h: Int): Double {
        val s = 160.0 / max(w, h)
        val cw = (w * s).toInt() + 2; val ch = (h * s).toInt() + 2
        val ma = Mat.zeros(ch, cw, CvType.CV_8UC1); val mb = Mat.zeros(ch, cw, CvType.CV_8UC1); val both = Mat()
        try {
            val pa = MatOfPoint(*a.map { Point(it.x * s, it.y * s) }.toTypedArray())
            val pb = MatOfPoint(*b.map { Point(it.x * s, it.y * s) }.toTypedArray())
            Imgproc.fillConvexPoly(ma, pa, org.opencv.core.Scalar(1.0)); Imgproc.fillConvexPoly(mb, pb, org.opencv.core.Scalar(1.0))
            pa.release(); pb.release()
            Core.bitwise_and(ma, mb, both); val inter = Core.countNonZero(both)
            Core.bitwise_or(ma, mb, both); val union = Core.countNonZero(both)
            return if (union > 0) inter.toDouble() / union else 0.0
        } finally {
            ma.release(); mb.release(); both.release()
        }
    }

    // ---------------- candidates ----------------

    private fun contourQuads(binary: Mat, area: Double): List<List<V>> {
        val contours = mutableListOf<MatOfPoint>()
        val hierarchy = Mat()
        val work = binary.clone()
        Imgproc.findContours(work, contours, hierarchy, Imgproc.RETR_LIST, Imgproc.CHAIN_APPROX_SIMPLE)
        work.release(); hierarchy.release()
        val withHull = contours.mapNotNull { c ->
            val idx = MatOfInt()
            Imgproc.convexHull(c, idx)
            val all = c.toArray()
            val hullPts = idx.toArray().map { all[it] }
            idx.release()
            if (hullPts.size < 4) return@mapNotNull null
            val hull = MatOfPoint2f(*hullPts.toTypedArray())
            val a = Imgproc.contourArea(hull)
            if (a > area * 0.06) hull to a else { hull.release(); null }
        }.sortedByDescending { it.second }
        contours.forEach { it.release() }

        val out = mutableListOf<List<V>>()
        withHull.forEachIndexed { i, (hull, _) ->
            if (i < 10) {
                val per = Imgproc.arcLength(hull, true)
                for (f in doubleArrayOf(0.01, 0.02, 0.03, 0.045, 0.06, 0.08)) {
                    val ap = MatOfPoint2f()
                    Imgproc.approxPolyDP(hull, ap, f * per, true)
                    val count = ap.total().toInt()
                    if (count == 4) out += ap.toArray().map { V(it.x, it.y) }
                    ap.release()
                    if (count <= 4) break
                }
            }
            hull.release()
        }
        return out
    }

    private fun lineQuads(edges: Mat, edgeBytes: ByteArray, img: Img): List<List<V>> {
        val w = img.w; val h = img.h
        val linesMat = Mat()
        Imgproc.HoughLinesP(
            edges, linesMat, 1.0, PI / 180, 40,
            0.12 * min(w, h), 0.03 * max(w, h),
        )
        val raw = mutableListOf<Seg>()
        if (linesMat.rows() > 0) {
            val data = IntArray(linesMat.rows() * 4)
            linesMat.get(0, 0, data)
            for (i in 0 until linesMat.rows()) {
                val a = V(data[i * 4].toDouble(), data[i * 4 + 1].toDouble())
                val b = V(data[i * 4 + 2].toDouble(), data[i * 4 + 3].toDouble())
                raw += Seg(a, b, (b - a).len(), atan2(b.y - a.y, b.x - a.x))
            }
        }
        linesMat.release()

        // 1. Join fragments of the same straight line. The line finder is randomised and often
        //    returns a page edge as several short pieces; joined, it competes as one long line.
        raw.sortByDescending { it.length }
        class Group(var a: V, var b: V, var covered: Double)
        val groups = mutableListOf<Group>()
        val joinDist = max(2.5, 0.006 * max(w, h))
        for (s in raw) {
            var joined = false
            for (g in groups) {
                val d = g.b - g.a; val len = d.len() + 1e-9
                val ux = d.x / len; val uy = d.y / len
                // Both ends of the piece must lie on the group's line.
                val da = abs(-(s.a.x - g.a.x) * uy + (s.a.y - g.a.y) * ux)
                val db = abs(-(s.b.x - g.a.x) * uy + (s.b.y - g.a.y) * ux)
                if (da > joinDist || db > joinDist) continue
                val ta = (s.a.x - g.a.x) * ux + (s.a.y - g.a.y) * uy
                val tb = (s.b.x - g.a.x) * ux + (s.b.y - g.a.y) * uy
                val lo = min(ta, tb); val hi = max(ta, tb)
                // Not across a big gap (two different objects that happen to line up).
                val gap = max(lo - len, -hi)
                if (gap > 0.25 * max(len, s.length)) continue
                val newLo = min(0.0, lo); val newHi = max(len, hi)
                val overlap = max(0.0, min(len, hi) - max(0.0, lo))
                g.covered += s.length - overlap * 0.999
                val na = V(g.a.x + ux * newLo, g.a.y + uy * newLo)
                val nb = V(g.a.x + ux * newHi, g.a.y + uy * newHi)
                g.a = na; g.b = nb
                joined = true
                break
            }
            if (!joined) groups += Group(s.a, s.b, s.length)
        }
        val joinedSegs = groups.map { g ->
            val extent = (g.b - g.a).len()
            Seg(g.a, g.b, min(extent, g.covered), atan2(g.b.y - g.a.y, g.b.x - g.a.x))
        }

        // 2. Drop near-duplicates (same direction, same place), longest first.
        val merged = mutableListOf<Seg>()
        for (s in joinedSegs.sortedByDescending { it.length }) {
            val dup = merged.any { m ->
                val dth = abs(((s.theta - m.theta) + PI / 2).mod(PI) - PI / 2)
                if (dth >= Math.toRadians(4.0)) false else {
                    val mid = (s.a + s.b) * 0.5
                    val d = m.b - m.a
                    val nx = -d.y / (d.len() + 1e-9); val ny = d.x / (d.len() + 1e-9)
                    abs((mid.x - m.a.x) * nx + (mid.y - m.a.y) * ny) < 0.02 * max(w, h)
                }
            }
            if (!dup) merged += s
        }
        // 3. Straighten each line against the real edge pixels (fixes small angle errors), measure how
        //    far it really runs (the line finder often reports only a piece of a page edge), and
        //    put page edges first: long lines with one side consistently different from the other.
        //    (Ranking by the reported length alone let floorboard or tile lines crowd out the page.)
        val measured = merged.map { measure(fitSegment(it, edgeBytes, img), img) }
        val ranked = mutableListOf<Seg>()
        for (s in measured.sortedByDescending { it.length * (0.15 + lineQuality(it, img)) }) {
            // The same edge can now appear twice (two pieces grown into the same full line).
            val dup = ranked.any { m ->
                val dth = abs(((s.theta - m.theta) + PI / 2).mod(PI) - PI / 2)
                if (dth >= Math.toRadians(3.0)) false else {
                    val mid = (s.a + s.b) * 0.5
                    val d = m.b - m.a
                    val nx = -d.y / (d.len() + 1e-9); val ny = d.x / (d.len() + 1e-9)
                    abs((mid.x - m.a.x) * nx + (mid.y - m.a.y) * ny) < 0.012 * max(w, h)
                }
            }
            if (!dup) ranked += s
        }

        val border = listOf(
            Seg(V(0.0, 0.0), V(w - 1.0, 0.0), w.toDouble(), 0.0),
            Seg(V(0.0, h - 1.0), V(w - 1.0, h - 1.0), w.toDouble(), 0.0),
            Seg(V(0.0, 0.0), V(0.0, h - 1.0), h.toDouble(), PI / 2),
            Seg(V(w - 1.0, 0.0), V(w - 1.0, h - 1.0), h.toDouble(), PI / 2),
        )
        val hor = (ranked.filter { abs(sin(it.theta)) < 0.7 } + border.take(2)).take(MAX_LINES)
        val ver = (ranked.filter { abs(sin(it.theta)) >= 0.7 } + border.drop(2)).take(MAX_LINES)

        if (trace) lineTrace = "raw=${raw.size} merged=${merged.size}\n" +
            "  slanted raw: " + raw.filter { val d = Math.toDegrees(it.theta).mod(90.0); d > 2 && d < 88 }.joinToString(" ") { "(%.0f,%.0f)-(%.0f,%.0f)".format(it.a.x, it.a.y, it.b.x, it.b.y) } + "\n" +
            "  ranked slanted: " + ranked.filter { val d = Math.toDegrees(it.theta).mod(90.0); d > 2 && d < 88 }.joinToString(" ") { "(%.0f,%.0f)-(%.0f,%.0f) len=%.0f q=%.2f".format(it.a.x, it.a.y, it.b.x, it.b.y, it.length, lineQuality(it, img)) } + "\n" +
            "  slanted merged: " + merged.filter { val d = Math.toDegrees(it.theta).mod(90.0); d > 2 && d < 88 }.joinToString(" ") { "(%.0f,%.0f)-(%.0f,%.0f)".format(it.a.x, it.a.y, it.b.x, it.b.y) } + "\n" + (hor.map { "H" to it } + ver.map { "V" to it }).joinToString("\n") { (k, sg) ->
            "  $k (%.0f,%.0f)-(%.0f,%.0f) len=%.0f q=%.2f".format(sg.a.x, sg.a.y, sg.b.x, sg.b.y, sg.length, lineQuality(sg, img))
        }
        val out = mutableListOf<List<V>>()
        for (i in hor.indices) for (j in i + 1 until hor.size) {
            for (a in ver.indices) for (b in a + 1 until ver.size) {
                val p1 = intersect(hor[i], ver[a]) ?: continue
                val p2 = intersect(hor[i], ver[b]) ?: continue
                val p3 = intersect(hor[j], ver[a]) ?: continue
                val p4 = intersect(hor[j], ver[b]) ?: continue
                out += listOf(p1, p2, p3, p4)
            }
        }
        return out
    }

    /** Re-fits a line to the edge pixels lying along it (running the same way), over 2x its length. */
    private fun fitSegment(s: Seg, edges: ByteArray, img: Img): Seg {
        val d = s.b - s.a
        val len = d.len()
        if (len < 1) return s
        val ux = d.x / len; val uy = d.y / len
        val nx = -uy; val ny = ux
        val pts = mutableListOf<Point>()
        var t = -0.5 * len
        while (t < 1.5 * len) {
            for (off in -4..4) {
                val x = (s.a.x + ux * t + nx * off).roundToInt()
                val y = (s.a.y + uy * t + ny * off).roundToInt()
                if (x !in 0 until img.w || y !in 0 until img.h) continue
                val i = y * img.w + x
                if (edges[i].toInt() == 0) continue
                if (abs(img.gx[i] * nx + img.gy[i] * ny) / (img.mag[i] + 1e-6) > 0.85) pts += Point(x.toDouble(), y.toDouble())
            }
            t += 1.0
        }
        if (pts.size < 20) return s
        val mp = MatOfPoint2f(*pts.toTypedArray())
        val line = Mat()
        Imgproc.fitLine(mp, line, Imgproc.DIST_HUBER, 0.0, 0.01, 0.01)
        val f = FloatArray(4); line.get(0, 0, f)
        mp.release(); line.release()
        val p0 = V(f[2].toDouble(), f[3].toDouble())
        val dv = V(f[0].toDouble(), f[1].toDouble())
        val ta = (s.a.x - p0.x) * dv.x + (s.a.y - p0.y) * dv.y
        val tb = (s.b.x - p0.x) * dv.x + (s.b.y - p0.y) * dv.y
        val a2 = p0 + dv * ta
        val b2 = p0 + dv * tb
        return Seg(a2, b2, s.length, atan2(b2.y - a2.y, b2.x - a2.x))
    }

    /**
     * How far a line really runs: walks along it across the whole picture and keeps the longest
     * stretch of edge pixels running the same way (small gaps allowed). The result's length is the
     * number of edge pixels on that stretch, so it no longer depends on how the line finder cut it.
     */
    private fun measure(s: Seg, img: Img): Seg {
        val d = s.b - s.a
        val len = d.len()
        if (len < 1) return s
        val ux = d.x / len; val uy = d.y / len
        val nx = -uy; val ny = ux
        val maxGap = max(6, (0.04 * img.maxDim).toInt())
        val reach = img.w + img.h
        var bestHits = 0; var bestStart = 0; var bestEnd = 0
        var hits = 0; var start = 0; var last = 0
        fun close() {
            if (hits > bestHits) { bestHits = hits; bestStart = start; bestEnd = last }
            hits = 0
        }
        for (t in -reach..reach) {
            val px = s.a.x + ux * t; val py = s.a.y + uy * t
            if (px < -2 || py < -2 || px > img.w + 1 || py > img.h + 1) { if (hits > 0) close(); continue }
            var hit = false
            for (off in -2..2) {
                val x = (px + nx * off).roundToInt(); val y = (py + ny * off).roundToInt()
                if (x !in 0 until img.w || y !in 0 until img.h) continue
                val i = y * img.w + x
                val m = img.mag[i]
                if (m > GRAD_T && abs(img.gx[i] * nx + img.gy[i] * ny) / (m + 1e-6) > 0.85) { hit = true; break }
            }
            if (hit) {
                if (hits == 0) start = t
                hits++; last = t
            } else if (hits > 0 && t - last > maxGap) close()
        }
        if (hits > 0) close()
        if (bestHits < 0.5 * s.length) return s
        val a = V(s.a.x + ux * bestStart, s.a.y + uy * bestStart)
        val b = V(s.a.x + ux * bestEnd, s.a.y + uy * bestEnd)
        return Seg(a, b, bestHits.toDouble(), s.theta)
    }

    private fun lineQuality(s: Seg, img: Img): Double {
        val d = s.b - s.a
        val len = d.len() + 1e-9
        val nrm = V(-d.y / len, d.x / len)
        val off = 0.015 * img.maxDim + 2
        var best = 0.0
        for (ch in img.chans.indices) {
            val r = bandDiff(img, ch, s.a, s.b, 20, off, nrm) ?: continue
            best = max(best, min(1.0, r.first / 14) * abs(r.second))
        }
        return best
    }

    /**
     * Compares samples just on one side of a segment with the other side.
     * Returns (mean absolute difference, mean sign of clear differences), or null if mostly off-image.
     */
    private fun bandDiff(img: Img, ch: Int, a: V, b: V, n: Int, off: Double, nrm: V, flip: Boolean = false): Pair<Double, Double>? {
        var count = 0
        var absSum = 0.0
        var signSum = 0.0
        for (k in 0 until n) {
            val t = (k + 0.5) / n
            val px = a.x + (b.x - a.x) * t
            val py = a.y + (b.y - a.y) * t
            val ix = (px + nrm.x * off).roundToInt(); val iy = (py + nrm.y * off).roundToInt()
            val ox = (px - nrm.x * off).roundToInt(); val oy = (py - nrm.y * off).roundToInt()
            if (ix !in 0 until img.w || iy !in 0 until img.h || ox !in 0 until img.w || oy !in 0 until img.h) continue
            var dl = (img.v(ch, ix, iy) - img.v(ch, ox, oy)).toDouble()
            if (flip) dl = -dl
            count++
            absSum += abs(dl)
            if (abs(dl) > 4) signSum += sign(dl)
        }
        if (count < 6) return null
        return absSum / count to signSum / count
    }

    private fun intersect(l1: Seg, l2: Seg): V? {
        val r = l1.b - l1.a
        val s = l2.b - l2.a
        val den = r.x * s.y - r.y * s.x
        if (abs(den) < 1e-9) return null
        val qp = l2.a - l1.a
        val t = (qp.x * s.y - qp.y * s.x) / den
        return l1.a + r * t
    }

    // ---------------- scoring ----------------

    /** Fraction of points along a→b sitting on a strong edge that runs the same way. */
    private fun support(img: Img, a: V, b: V, n: Int = 32): Double {
        val d = b - a
        val len = d.len()
        if (len < 1) return 0.0
        val nx = -d.y / len; val ny = d.x / len
        var hits = 0
        for (k in 0 until n) {
            val t = (k + 0.5) / n
            val px = a.x + d.x * t; val py = a.y + d.y * t
            for (off in -2..2) {
                val x = (px + nx * off).roundToInt(); val y = (py + ny * off).roundToInt()
                if (x !in 0 until img.w || y !in 0 until img.h) continue
                val i = y * img.w + x
                val m = img.mag[i]
                if (m > GRAD_T && abs(img.gx[i] * nx + img.gy[i] * ny) / (m + 1e-6) > 0.85) { hits++; break }
            }
        }
        return hits.toDouble() / n
    }

    private fun score(q: List<V>, img: Img, parts: StringBuilder? = null): Double {
        val w = img.w; val h = img.h
        if (!convex(q)) return 0.0
        val area = polygonArea(q)
        val r = area / (w.toDouble() * h)
        if (r < 0.08 || r > 0.995) return 0.0
        val minSide = min(w, h) * 0.12
        for (i in 0 until 4) {
            if ((q[(i + 1) % 4] - q[i]).len() < minSide) return 0.0
            val an = angle(q[(i + 3) % 4], q[i], q[(i + 1) % 4])
            if (an < 45 || an > 135) return 0.0
        }
        if (q.any { it.x < -0.03 * w || it.y < -0.03 * h || it.x > 1.03 * w || it.y > 1.03 * h }) return 0.0

        // Sides on real edges (sides along the photo border get a modest fixed value).
        val sups = DoubleArray(4)
        var borders = 0
        for (i in 0 until 4) {
            val a = q[i]; val b = q[(i + 1) % 4]
            val onBorder = listOf(a, b).all { p -> min(min(p.x, p.y), min(w - 1 - p.x, h - 1 - p.y)) < 3 }
            if (onBorder) borders++
            sups[i] = if (onBorder) 0.35 else support(img, a, b)
            if (sups[i] < 0.30) return 0.0
        }
        if (borders > 1) return 0.0

        // Paper inside, table outside, consistently along each side.
        val c = V(q.sumOf { it.x } / 4, q.sumOf { it.y } / 4)
        val off = 0.015 * img.maxDim + 2
        val seps = DoubleArray(4)
        for (i in 0 until 4) {
            val a = q[i]; val b = q[(i + 1) % 4]; val d = b - a; val len = d.len()
            var nrm = V(-d.y / len, d.x / len)
            val mid = (a + b) * 0.5
            if ((c.x - mid.x) * nrm.x + (c.y - mid.y) * nrm.y < 0) nrm = nrm * -1.0 // point inwards
            var bestSep: Double? = null
            for (ch in img.chans.indices) {
                val res = bandDiff(img, ch, a, b, 24, off, nrm, flip = ch == 1)
                if (res == null) { if (bestSep == null) bestSep = 0.3; continue }
                val sc = min(1.0, res.first / 14) * max(0.0, res.second)
                bestSep = if (bestSep == null) sc else max(bestSep, sc)
            }
            seps[i] = bestSep ?: 0.3
        }
        // A crisp continuous edge with similar brightness on both sides (page on another white sheet)
        // is still allowed, with a low value.
        for (i in 0 until 4) if (seps[i] < 0.15 && sups[i] >= 0.9) seps[i] = 0.3
        if (seps.min() < 0.15) return 0.0

        // Real corners: the edge must not carry on past them.
        val ext = DoubleArray(8)
        for (i in 0 until 4) {
            val a = q[i]; val b = q[(i + 1) % 4]; val d = b - a
            ext[i * 2] = support(img, b + d * 0.03, b + d * 0.18, 12)
            ext[i * 2 + 1] = support(img, a - d * 0.03, a - d * 0.18, 12)
        }
        ext.sort()
        val extPenalty = 1 - 0.6 * (ext[6] + ext[7]) / 2

        // Overall: paper brighter than just outside.
        val inner = q.map { c + (it - c) * 0.94 }
        val outer = q.map { c + (it - c) * 1.06 }
        var diffSum = 0.0; var diffN = 0
        for (i in 0 until 4) {
            for (k in 1..9) {
                val t = 0.1 + 0.8 * (k - 1) / 8
                val pi = inner[i] + (inner[(i + 1) % 4] - inner[i]) * t
                val po = outer[i] + (outer[(i + 1) % 4] - outer[i]) * t
                val xi = pi.x.roundToInt(); val yi = pi.y.roundToInt()
                val xo = po.x.roundToInt(); val yo = po.y.roundToInt()
                if (xi in 0 until w && yi in 0 until h && xo in 0 until w && yo in 0 until h) {
                    diffSum += img.v(0, xi, yi) - img.v(0, xo, yo); diffN++
                }
            }
        }
        val pol = if (diffN < 8) 1.0 else 1 + (diffSum / diffN / 255).coerceIn(-0.25, 0.3) * 1.2

        // Prefer the sheet with printing on it over a blank one.
        var inked = 0; var total = 0
        for (i in 2..18) for (j in 2..18) {
            val u = i / 20.0; val v = j / 20.0
            val top = q[0] + (q[1] - q[0]) * u; val bot = q[3] + (q[2] - q[3]) * u
            val p = top + (bot - top) * v
            val x = p.x.roundToInt(); val y = p.y.roundToInt()
            if (x in 0 until w && y in 0 until h) { total++; if (img.text[y * w + x].toInt() != 0) inked++ }
        }
        val density = inked.toDouble() / max(total, 1)
        val printing = 0.55 + 0.45 * min(1.0, density / 0.06)

        val supMean = sups.average(); val sepMean = seps.average()
        if (parts != null) parts.append("r=%.3f sups=%s seps=%s pol=%.3f ext=%.3f print=%.3f".format(
            r, sups.joinToString(",") { "%.2f".format(it) }, seps.joinToString(",") { "%.2f".format(it) }, pol, extPenalty, printing))
        return sqrt(r) * supMean * supMean * sqrt(sups.min()) * pol * sepMean * sqrt(seps.min()) * extPenalty * printing
    }

    private fun scoreParts(q: List<V>, img: Img): String = StringBuilder().also { score(q, img, it) }.toString()

    // ---------------- refine + inset ----------------

    /** Fits a straight line to the edge pixels along each side, then intersects neighbouring sides. */
    private fun refine(q: List<V>, img: Img, edges: ByteArray): List<V> {
        val w = img.w; val h = img.h
        val band = 0.012 * img.maxDim + 2
        val lines = mutableListOf<Seg>()
        for (i in 0 until 4) {
            val a = q[i]; val b = q[(i + 1) % 4]; val d = b - a; val len = d.len()
            val ux = d.x / len; val uy = d.y / len
            val nx = -uy; val ny = ux
            val minX = max(0, (min(a.x, b.x) - band).toInt()); val maxX = min(w - 1, (max(a.x, b.x) + band).toInt())
            val minY = max(0, (min(a.y, b.y) - band).toInt()); val maxY = min(h - 1, (max(a.y, b.y) + band).toInt())
            val pts = mutableListOf<Point>()
            for (y in minY..maxY) for (x in minX..maxX) {
                if (edges[y * w + x].toInt() == 0) continue
                // Only edge pixels running the same way as this side (ignores wood grain, text, patterns).
                val gi = y * w + x
                if (abs(img.gx[gi] * nx + img.gy[gi] * ny) / (img.mag[gi] + 1e-6) <= 0.85) continue
                val rx = x - a.x; val ry = y - a.y
                val along = rx * ux + ry * uy
                val dist = rx * nx + ry * ny
                if (abs(dist) < band && along > 0.05 * len && along < 0.95 * len) pts += Point(x.toDouble(), y.toDouble())
            }
            if (pts.size < 15) { lines += Seg(a, b, len, 0.0); continue }
            val mp = MatOfPoint2f(*pts.toTypedArray())
            val line = Mat()
            Imgproc.fitLine(mp, line, Imgproc.DIST_HUBER, 0.0, 0.01, 0.01)
            val f = FloatArray(4); line.get(0, 0, f)
            mp.release(); line.release()
            val p0 = V(f[2].toDouble(), f[3].toDouble())
            lines += Seg(p0, p0 + V(f[0].toDouble(), f[1].toDouble()), 1.0, 0.0)
        }
        val out = mutableListOf<V>()
        for (i in 0 until 4) out += intersect(lines[(i + 3) % 4], lines[i]) ?: return q
        val maxMove = (0 until 4).maxOf { (out[it] - q[it]).len() }
        return if (convex(out) && maxMove < 0.02 * img.maxDim) out else q
    }

    /** Moves every side inwards a little, so the crop sits just inside the page. */
    private fun inset(q: List<V>): List<V> {
        val widthPx = ((q[1] - q[0]).len() + (q[2] - q[3]).len()) / 2
        val heightPx = ((q[3] - q[0]).len() + (q[2] - q[1]).len()) / 2
        val d = INSET * min(widthPx, heightPx)
        val c = V(q.sumOf { it.x } / 4, q.sumOf { it.y } / 4)
        val shifted = (0 until 4).map { i ->
            val a = q[i]; val b = q[(i + 1) % 4]; val v = b - a; val len = v.len()
            var nrm = V(-v.y / len, v.x / len)
            val mid = (a + b) * 0.5
            if ((c.x - mid.x) * nrm.x + (c.y - mid.y) * nrm.y < 0) nrm = nrm * -1.0
            Seg(a + nrm * d, b + nrm * d, len, 0.0)
        }
        val out = mutableListOf<V>()
        for (i in 0 until 4) out += intersect(shifted[(i + 3) % 4], shifted[i]) ?: return q
        return if (convex(out)) out else q
    }

    // ---------------- geometry helpers ----------------

    /** TL, TR, BR, BL order (null if the points are degenerate). */
    private fun order(pts: List<V>): List<V>? {
        if (pts.size != 4) return null
        val tl = pts.minBy { it.x + it.y }; val br = pts.maxBy { it.x + it.y }
        val tr = pts.maxBy { it.x - it.y }; val bl = pts.minBy { it.x - it.y }
        val q = listOf(tl, tr, br, bl)
        if (q.toSet().size == 4) return q
        val cx = pts.sumOf { it.x } / 4; val cy = pts.sumOf { it.y } / 4
        return pts.sortedBy { atan2(it.y - cy, it.x - cx) }
    }

    private fun convex(q: List<V>): Boolean {
        var s = 0
        for (i in 0 until 4) {
            val a = q[i]; val b = q[(i + 1) % 4]; val c = q[(i + 2) % 4]
            val cr = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if (abs(cr) < 1e-6) return false
            val sg = if (cr > 0) 1 else -1
            if (s == 0) s = sg else if (sg != s) return false
        }
        return true
    }

    private fun polygonArea(p: List<V>): Double {
        var s = 0.0
        for (i in p.indices) { val a = p[i]; val b = p[(i + 1) % p.size]; s += a.x * b.y - b.x * a.y }
        return abs(s) / 2
    }

    private fun angle(a: V, b: V, c: V): Double {
        val v1 = a - b; val v2 = c - b
        val d = v1.len() * v2.len()
        if (d == 0.0) return 0.0
        return Math.toDegrees(acos(((v1.x * v2.x + v1.y * v2.y) / d).coerceIn(-1.0, 1.0)))
    }
}
