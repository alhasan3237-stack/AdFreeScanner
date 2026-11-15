package com.myscanner.app.scan

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Matrix
import android.net.Uri
import android.os.SystemClock
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.exifinterface.media.ExifInterface
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import kotlin.math.abs
import kotlin.math.max

/**
 * Looks at the camera picture a few times a second and reports where the page is, so the outline
 * can be drawn live. Uses a small COLOUR copy and the page networks (same checks as for photos),
 * and only moves the outline when a new position is seen twice in a row, so it doesn't flicker.
 *
 * It is careful with the phone's processor, so the camera picture and the buttons stay smooth:
 * it rests at least as long as it worked, the slow corner network rests while the page is steady
 * (its last answer is reused for up to a second), and it stops completely while a photo is taken.
 */
class EdgeAnalyzer(private val onResult: (Quad?) -> Unit) : ImageAnalysis.Analyzer {

    /** True while a photo is being taken or its animation plays: camera frames are skipped. */
    @Volatile var paused = false

    private var lastRun = 0L
    private var interval = 150L
    private var nv21 = ByteArray(0)
    private var smoothed: Quad? = null
    private var pending: Quad? = null
    private var misses = 0
    private var hint: DocQuad.Result? = null
    private var hintAt = 0L
    private var lastRaw: Quad? = null

    override fun analyze(image: ImageProxy) {
        val now = SystemClock.elapsedRealtime()
        if (paused || now - lastRun < interval) {
            image.close()
            return
        }
        lastRun = now
        val yuv = Mat()
        val bgr = Mat()
        val small = Mat()
        NativeGuard.enter(NativeGuard.LIVE)
        try {
            val w = image.width
            val h = image.height
            toNv21(image)
            yuv.create(h + h / 2, w, CvType.CV_8UC1)
            yuv.put(0, 0, nv21)
            Imgproc.cvtColor(yuv, bgr, Imgproc.COLOR_YUV2BGR_NV21)
            val s = LIVE_SIDE / max(w, h)
            Imgproc.resize(bgr, small, Size(w * s, h * s), 0.0, 0.0, Imgproc.INTER_AREA)
            ImageProcessor.rotateInPlace(small, image.imageInfo.rotationDegrees)
            liveFind(small, now)
        } catch (e: Throwable) {
            report(null)
        } finally {
            NativeGuard.leave(NativeGuard.LIVE)
            yuv.release()
            bgr.release()
            small.release()
            image.close()
            // Rest at least as long as the work took (slow phones look less often, and stay smooth).
            interval = (SystemClock.elapsedRealtime() - now).coerceIn(130, 450)
        }
    }

    private fun liveFind(small: Mat, now: Long) {
        run {
            if (hint == null || now - hintAt > HINT_MS) { hint = DocQuad.run(small); hintAt = now }
            val q = DocumentDetector.detect(small, hint)
            // The picture is changing (the phone or the page moved): ask the corner network again next time.
            val before = lastRaw
            if (q == null || before == null || farApart(before, q, 0.03f)) hintAt = 0
            lastRaw = q
            report(q)
        }
    }

    private companion object {
        /** Long side of the live picture that is searched. */
        const val LIVE_SIDE = 400.0
        /** How long one answer of the corner network is reused. */
        const val HINT_MS = 900L
    }

    /** Copies the camera's YUV_420_888 frame into [nv21] (Y plane, then V/U interleaved). */
    private fun toNv21(image: ImageProxy) {
        val w = image.width
        val h = image.height
        if (nv21.size != w * h + w * (h / 2)) nv21 = ByteArray(w * h + w * (h / 2))
        val y = image.planes[0]
        val yBuf = y.buffer
        var pos = 0
        for (row in 0 until h) {
            yBuf.position(row * y.rowStride)
            yBuf.get(nv21, pos, w)
            pos += w
        }
        val u = image.planes[1]
        val v = image.planes[2]
        val uBuf = u.buffer
        val vBuf = v.buffer
        val cw = w / 2
        val ch = h / 2
        for (row in 0 until ch) {
            for (col in 0 until cw) {
                val vi = row * v.rowStride + col * v.pixelStride
                val ui = row * u.rowStride + col * u.pixelStride
                if (pos + 1 >= nv21.size || vi >= vBuf.limit() || ui >= uBuf.limit()) return
                nv21[pos++] = vBuf.get(vi)
                nv21[pos++] = uBuf.get(ui)
            }
        }
    }

    /**
     * Steadies the outline: small moves are smoothed, a big jump is only accepted when the next
     * frame confirms it, and the outline hides only after several misses in a row.
     */
    private fun report(q: Quad?) {
        if (q == null) {
            misses++
            if (misses >= 4) { smoothed = null; pending = null }
            onResult(smoothed)
            return
        }
        misses = 0
        val prev = smoothed
        smoothed = when {
            prev == null -> {
                // First sighting: show it only once a second frame agrees.
                val p = pending
                pending = q
                if (p != null && !farApart(p, q, 0.06f)) q else null
            }
            !farApart(prev, q, 0.12f) -> { pending = null; lerp(prev, q, 0.6f) }
            else -> {
                val p = pending
                pending = q
                if (p != null && !farApart(p, q, 0.06f)) q else prev
            }
        }
        onResult(smoothed)
    }

    private fun farApart(a: Quad, b: Quad, limit: Float): Boolean =
        a.points.zip(b.points).any { (p, q) -> abs(p.x - q.x) > limit || abs(p.y - q.y) > limit }

    private fun lerp(a: Quad, b: Quad, t: Float): Quad {
        val pts = a.points.zip(b.points).map { (p, q) -> P(p.x + (q.x - p.x) * t, p.y + (q.y - p.y) * t) }
        return Quad(pts[0], pts[1], pts[2], pts[3])
    }
}

/**
 * Copies a gallery photo into [dest] as an upright JPEG no bigger than [maxSide].
 * Handles HEIC/PNG/WebP and rotated phone photos.
 */
fun importGalleryImage(context: Context, uri: Uri, dest: File, maxSide: Int = ImageProcessor.MAX_SIDE) {
    val resolver = context.contentResolver
    val degrees = try {
        resolver.openInputStream(uri)?.use { ExifInterface(it).rotationDegrees } ?: 0
    } catch (e: Exception) {
        0
    }
    val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
    resolver.openInputStream(uri)?.use { BitmapFactory.decodeStream(it, null, bounds) }
    val longest = max(bounds.outWidth, bounds.outHeight)
    if (longest <= 0) error("Not a supported image")

    var sample = 1
    while (longest / (sample * 2) >= maxSide) sample *= 2
    val decoded = resolver.openInputStream(uri)?.use {
        BitmapFactory.decodeStream(it, null, BitmapFactory.Options().apply { inSampleSize = sample })
    } ?: error("Could not open the image")

    val scale = minOf(1f, maxSide.toFloat() / max(decoded.width, decoded.height))
    val matrix = Matrix().apply {
        if (scale < 1f) postScale(scale, scale)
        if (degrees != 0) postRotate(degrees.toFloat())
    }
    val upright = if (matrix.isIdentity) decoded else
        Bitmap.createBitmap(decoded, 0, 0, decoded.width, decoded.height, matrix, true)
    dest.outputStream().use { upright.compress(Bitmap.CompressFormat.JPEG, 95, it) }
    if (upright !== decoded) upright.recycle()
    decoded.recycle()
}
