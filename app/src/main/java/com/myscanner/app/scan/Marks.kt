package com.myscanner.app.scan

import java.util.Base64

/**
 * Something drawn on top of a page: a pen or highlighter stroke, or a piece of typed text
 * (a signature is just pen strokes). Positions are 0..1 across the page as it is shown
 * (after rotation), so they fit any size the page is rendered at.
 */
sealed class Mark {
    /**
     * @param width line thickness as a fraction of the page's shorter side
     * @param highlighter true = see-through marker that never hides the text under it
     */
    data class Stroke(val points: List<P>, val color: Int, val width: Float, val highlighter: Boolean) : Mark()

    /**
     * @param x,y the centre of the text
     * @param size letter height as a fraction of the page's shorter side
     * @param angle clockwise turn in degrees (0, 90, 180, 270): keeps the text upright with the
     *   content when the page is rotated later
     */
    data class Label(val text: String, val x: Float, val y: Float, val size: Float, val color: Int, val angle: Int = 0) : Mark()
}

/** Saving, loading and rotating marks. */
object Marks {

    /** One mark per line: "S|colour|width|highlighter|x,y;x,y;..." or "T|colour|size|angle|x|y|base64 text". */
    fun encode(marks: List<Mark>): String = marks.joinToString("\n") { m ->
        when (m) {
            is Mark.Stroke -> "S|${m.color}|${m.width}|${if (m.highlighter) 1 else 0}|" +
                m.points.joinToString(";") { "%.4f,%.4f".format(java.util.Locale.US, it.x, it.y) }
            is Mark.Label -> "T|${m.color}|${m.size}|${m.angle}|${m.x}|${m.y}|" +
                Base64.getEncoder().encodeToString(m.text.toByteArray(Charsets.UTF_8))
        }
    }

    /** Reads [encode]'s text; lines that can't be read are skipped. */
    fun decode(text: String?): List<Mark> {
        if (text.isNullOrBlank()) return emptyList()
        return text.split("\n").mapNotNull { line ->
            runCatching {
                val f = line.split("|")
                when (f[0]) {
                    "S" -> Mark.Stroke(
                        points = f[4].split(";").filter { it.isNotEmpty() }.map { p -> p.split(",").let { P(it[0].toFloat(), it[1].toFloat()) } },
                        color = f[1].toInt(), width = f[2].toFloat(), highlighter = f[3] == "1",
                    ).takeIf { it.points.isNotEmpty() }
                    "T" -> Mark.Label(
                        text = String(Base64.getDecoder().decode(f[6]), Charsets.UTF_8),
                        x = f[4].toFloat(), y = f[5].toFloat(), size = f[2].toFloat(), color = f[1].toInt(), angle = f[3].toInt(),
                    )
                    else -> null
                }
            }.getOrNull()
        }
    }

    /** Where the marks end up when the page is turned clockwise by [degrees] (a multiple of 90). */
    fun rotate(marks: List<Mark>, degrees: Int): List<Mark> {
        val d = ((degrees % 360) + 360) % 360
        if (d == 0) return marks
        fun turn(p: P): P = when (d) {
            90 -> P(1f - p.y, p.x)
            180 -> P(1f - p.x, 1f - p.y)
            else -> P(p.y, 1f - p.x)
        }
        return marks.map { m ->
            when (m) {
                is Mark.Stroke -> m.copy(points = m.points.map(::turn))
                is Mark.Label -> turn(P(m.x, m.y)).let { c -> m.copy(x = c.x, y = c.y, angle = (m.angle + d) % 360) }
            }
        }
    }
}
