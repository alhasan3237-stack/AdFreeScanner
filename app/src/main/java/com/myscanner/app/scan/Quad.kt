package com.myscanner.app.scan

import kotlin.math.abs

/** A point in normalised coordinates: 0..1 across the upright image. */
data class P(val x: Float, val y: Float)

/**
 * The four corners of a page, in normalised coordinates, always in the order
 * top-left, top-right, bottom-right, bottom-left.
 */
data class Quad(val tl: P, val tr: P, val br: P, val bl: P) {

    val points: List<P> get() = listOf(tl, tr, br, bl)

    fun with(index: Int, p: P): Quad = when (index) {
        0 -> copy(tl = p)
        1 -> copy(tr = p)
        2 -> copy(br = p)
        else -> copy(bl = p)
    }

    /** True when the corners make a proper (non-crossed) four-sided shape. */
    fun isConvex(): Boolean {
        val pts = points
        var sign = 0
        for (i in 0 until 4) {
            val a = pts[i]
            val b = pts[(i + 1) % 4]
            val c = pts[(i + 2) % 4]
            val cross = (b.x - a.x) * (c.y - b.y) - (b.y - a.y) * (c.x - b.x)
            if (abs(cross) < 1e-6f) return false
            val s = if (cross > 0) 1 else -1
            if (sign == 0) sign = s else if (s != sign) return false
        }
        return true
    }

    /** Text form for saving: "x,y;x,y;x,y;x,y". */
    fun encode(): String = points.joinToString(";") { "${it.x},${it.y}" }

    /** Short stable text used as a cache key. */
    fun key(): String = points.joinToString(",") { "%.4f:%.4f".format(it.x, it.y) }

    companion object {
        /** The whole image with a tiny margin: used when no page edge is found. */
        val FULL = Quad(P(0.02f, 0.02f), P(0.98f, 0.02f), P(0.98f, 0.98f), P(0.02f, 0.98f))

        /** Exactly the whole image (for pages the app made itself, like an ID card sheet). */
        val WHOLE = Quad(P(0f, 0f), P(1f, 0f), P(1f, 1f), P(0f, 1f))

        /** Reads [encode]'s text form; null if it isn't valid. */
        fun decode(text: String?): Quad? = runCatching {
            val pts = text!!.split(";").map { pair -> pair.split(",").let { P(it[0].toFloat(), it[1].toFloat()) } }
            if (pts.size != 4) null else Quad(pts[0], pts[1], pts[2], pts[3])
        }.getOrNull()

        /** Puts four points in top-left, top-right, bottom-right, bottom-left order. */
        fun ordered(points: List<P>): Quad {
            require(points.size == 4)
            val tl = points.minBy { it.x + it.y }
            val br = points.maxBy { it.x + it.y }
            val tr = points.maxBy { it.x - it.y }
            val bl = points.minBy { it.x - it.y }
            // Degenerate shapes can pick the same point twice: fall back to angle order.
            if (setOf(tl, tr, br, bl).size < 4) {
                val cx = points.map { it.x }.average().toFloat()
                val cy = points.map { it.y }.average().toFloat()
                val sorted = points.sortedBy { kotlin.math.atan2((it.y - cy).toDouble(), (it.x - cx).toDouble()) }
                // atan2 order starting from the left goes: TL (-135°), TR (-45°), BR (45°), BL (135°)
                return Quad(sorted[0], sorted[1], sorted[2], sorted[3])
            }
            return Quad(tl, tr, br, bl)
        }
    }
}
