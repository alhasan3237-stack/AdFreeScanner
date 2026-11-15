package com.myscanner.app.scan

import android.content.Context
import android.util.Log
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.opencv.core.Size
import org.opencv.dnn.Dnn
import org.opencv.dnn.Net
import org.opencv.imgproc.Imgproc
import java.io.File

/**
 * A small neural network (U-Net, ~350 KB) trained on 12,000 generated photos of documents on all
 * kinds of surfaces, with hands, clutter, shadows, glare and sheets under sheets. It marks which
 * pixels belong to the page. The edge detector uses it to propose outlines and to reject wrong ones.
 * Runs offline with OpenCV's built-in DNN module (no extra libraries).
 */
object PageNet {
    const val W = 96
    const val H = 128

    @Volatile private var net: Net? = null

    val available: Boolean get() = net != null
    private val lock = Any()

    /** Loads the network (once). Slow-ish, so the app calls it on a background thread at start-up. */
    @Synchronized
    fun init(context: Context) {
        if (net != null) return
        runCatching {
            val file = File(context.filesDir, "pagenet.onnx")
            context.assets.open("pagenet.onnx").use { input -> file.outputStream().use { input.copyTo(it) } }
            net = Dnn.readNetFromONNX(file.path)
        }.onFailure { Log.e("PageNet", "Could not load the page network", it) }
    }

    /**
     * Page probability (0..1) for each pixel of a [H] x [W] grid covering the upright image, as a
     * single-channel float Mat. Null if the network isn't available.
     */
    /** Turned off if it ever crashed on this phone (see [NativeGuard]). */
    @Volatile var enabled = true

    fun predict(image: Mat): Mat? = synchronized(lock) { NativeGuard.guard(NativeGuard.PAGE_NET) { predictInner(image) } }

    private fun predictInner(image: Mat): Mat? {
        val n = net ?: return null
        if (!enabled) return null
        return runCatching {
            val bgr = if (image.channels() == 1) Mat().also { Imgproc.cvtColor(image, it, Imgproc.COLOR_GRAY2BGR) } else image
            val blob = Dnn.blobFromImage(bgr, 1.0, Size(W.toDouble(), H.toDouble()), Scalar(0.0, 0.0, 0.0), false, false)
            if (bgr !== image) bgr.release()
            n.setInput(blob)
            val out = n.forward()
            val prob = out.reshape(1, H).clone() // 1x1xHxW -> H rows x W columns
            blob.release(); out.release()
            prob
        }.getOrNull()
    }
}
