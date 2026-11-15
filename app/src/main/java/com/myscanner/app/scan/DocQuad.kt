package com.myscanner.app.scan

import android.content.Context
import android.util.Log
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Rect
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.exp
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

/**
 * The page-corner network (MobileNetV3 + FPN, 256x256 input). Started from the open-source
 * MakeACopy scanner's model (Apache-2.0, trained on thousands of real document photos), then
 * trained further on 24,000 realistic scenes: carpets, bedsheets, tiles, notebooks, receipts,
 * fingers, shadows, glare and dim light. Runs offline in OpenCV's DNN module.
 *
 * Gives the four page corners and how sure it is; the edge detector combines that with the real edges.
 */
object DocQuad {
    private const val IN = 256
    private const val OUT = 64

    @Volatile private var net: Net? = null
    private val lock = Any()

    class Result(
        /** Corners as fractions (0..1) of the picture's width and height: TL, TR, BR, BL. */
        val corners: List<Pair<Double, Double>>,
        /** How sure the network is of its weakest corner (0..1). */
        val confidence: Double,
    )

    /** Loads the network (once). Takes a moment (13 MB), so the app calls it on a background thread at start-up. */
    @Synchronized
    fun init(context: Context) {
        if (net != null) return
        runCatching {
            val file = File(context.filesDir, "docquad.onnx")
            // Copy once per app version (the asset is 13 MB).
            val stamp = File(context.filesDir, "docquad.version")
            val version = context.packageManager.getPackageInfo(context.packageName, 0).lastUpdateTime.toString()
            if (!file.exists() || !stamp.exists() || stamp.readText() != version) {
                context.assets.open("docquad.onnx").use { input -> file.outputStream().use { input.copyTo(it) } }
                stamp.writeText(version)
            }
            net = Dnn.readNetFromONNX(file.path)
        }.onFailure { Log.e("DocQuad", "Could not load the corner network", it) }
    }

    val available: Boolean get() = net != null

    /** Turned off in the camera's safe mode (after the camera closed unexpectedly). */
    @Volatile var enabled = true

    private fun sigmoid(x: Float): Double = 1.0 / (1.0 + exp(-x.toDouble()))

    /** [image]: BGR (or grey) picture of any size. Null if the network isn't available. */
    fun run(image: Mat): Result? = synchronized(lock) { NativeGuard.guard(NativeGuard.CORNER_NET) { runInner(image) } }

    private fun runInner(image: Mat): Result? {
        val n = net ?: return null
        if (!enabled) return null
        val bgr = if (image.channels() == 1) Mat().also { Imgproc.cvtColor(image, it, Imgproc.COLOR_GRAY2BGR) } else image
        val w = bgr.cols(); val h = bgr.rows()
        val s = min(IN.toDouble() / w, IN.toDouble() / h)
        val nw = max(1, (w * s).roundToInt()); val nh = max(1, (h * s).roundToInt())
        val ox = (IN - nw) / 2; val oy = (IN - nh) / 2
        val box = Mat.zeros(IN, IN, CvType.CV_8UC3) // black bars, exactly as in training
        val small = Mat()
        val outs = ArrayList<Mat>()
        var blob: Mat? = null
        return try {
            Imgproc.resize(bgr, small, Size(nw.toDouble(), nh.toDouble()), 0.0, 0.0, Imgproc.INTER_AREA)
            small.copyTo(box.submat(Rect(ox, oy, nw, nh)))
            blob = Dnn.blobFromImage(box, 1.0 / 255.0, Size(IN.toDouble(), IN.toDouble()), Scalar(0.0, 0.0, 0.0), true, false)
            n.setInput(blob)
            // Both outputs are asked for, exactly as in version 7 (which is proven on phones),
            // even though only the corners are used.
            n.forward(outs, listOf("corner_heatmaps", "mask_logits"))
            val hm = FloatArray(4 * OUT * OUT); outs[0].reshape(1, 1).get(0, 0, hm)

            // Corners: peak of each heatmap, refined by a 3x3 weighted centre.
            val corners = ArrayList<Pair<Double, Double>>(4)
            var conf = 1.0
            for (c in 0 until 4) {
                val off = c * OUT * OUT
                var best = 0
                for (i in 1 until OUT * OUT) if (hm[off + i] > hm[off + best]) best = i
                val iy = best / OUT; val ix = best % OUT
                conf = min(conf, sigmoid(hm[off + best]))
                var sw = 0.0; var sx = 0.0; var sy = 0.0
                for (y in max(0, iy - 1)..min(OUT - 1, iy + 1)) for (x in max(0, ix - 1)..min(OUT - 1, ix + 1)) {
                    val wt = sigmoid(hm[off + y * OUT + x]); sw += wt; sx += wt * (x + 0.5); sy += wt * (y + 0.5)
                }
                val cx = sx / sw * 4.0; val cy = sy / sw * 4.0
                corners += ((cx - ox) / nw) to ((cy - oy) / nh)
            }
            Result(corners, conf)
        } catch (e: Throwable) {
            Log.e("DocQuad", "Corner network failed", e)
            null
        } finally {
            blob?.release(); small.release(); box.release(); outs.forEach { it.release() }
            if (bgr !== image) bgr.release()
        }
    }
}
