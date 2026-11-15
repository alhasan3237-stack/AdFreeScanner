package com.myscanner.app.ui.scan

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.CubicBezierEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.snapshotFlow
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.geometry.lerp
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.myscanner.app.scan.Quad
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.math.min
import kotlin.math.roundToInt

/** What the auto-enhance animation shows: the photo, the page outline, then the flat and enhanced page. */
data class EnhanceAnim(
    val key: Long,
    /** The picture frozen at the moment of the shot. */
    val photo: ImageBitmap,
    /** The page outline, once the photo has been checked. */
    val quad: Quad? = null,
    val flat: ImageBitmap? = null,
    val enhanced: ImageBitmap? = null,
    /** Preparing the page failed: end the animation quietly. */
    val failed: Boolean = false,
)

private val Accent = Color(0xFF2EC4B6)

/** Gentle ease-in-out: starts and ends softly, no sudden jumps. */
private val Smooth = CubicBezierEasing(0.45f, 0f, 0.25f, 1f)

/** Nearly even speed with soft ends, for the scan line sweeping down the page. */
private val Glide = CubicBezierEasing(0.3f, 0f, 0.4f, 1f)

/**
 * The auto-enhance animation played after each photo (about 2.9 seconds; tap the shutter to skip it):
 *  0. the picture freezes at the shot, a light sweeps over it while the page is found,
 *  1. the outline snaps from the photo's edges onto the page,
 *  2. the page lifts out of the photo and straightens into a flat sheet,
 *  3. a glowing scan line sweeps down, revealing the cleaned-up page,
 *  4. the page shrinks away towards the page counter.
 */
@Composable
fun EnhanceOverlay(anim: EnhanceAnim, onFinished: () -> Unit) {
    val current by rememberUpdatedState(anim)
    val done by rememberUpdatedState(onFinished)
    val outline = remember(anim.key) { Animatable(0f) }
    val lift = remember(anim.key) { Animatable(0f) }
    val scan = remember(anim.key) { Animatable(0f) }
    val away = remember(anim.key) { Animatable(0f) }

    // While the photo is being checked: a soft light sweeps over the frozen picture.
    val sweep by rememberInfiniteTransition(label = "sweep").animateFloat(
        0f, 1f, infiniteRepeatable(tween(1500, easing = LinearEasing), RepeatMode.Restart), label = "sweepPos",
    )

    LaunchedEffect(anim.key) {
        // Wait for the page outline (the photo is being saved and checked; never more than 8 s).
        withTimeoutOrNull(8000) { snapshotFlow { current.quad != null || current.failed }.first { it } }
        if (current.quad == null) { done(); return@LaunchedEffect }
        outline.animateTo(1f, tween(520, easing = Smooth))
        delay(120)
        // Wait for the straightened page (normally ready by now; never more than 5 s).
        withTimeoutOrNull(5000) { snapshotFlow { current.flat != null || current.failed }.first { it } }
        if (current.flat == null) { away.animateTo(1f, tween(450, easing = Smooth)); done(); return@LaunchedEffect }
        lift.animateTo(1f, tween(640, easing = Smooth))
        delay(80)
        if (current.enhanced != null) scan.animateTo(1f, tween(1000, easing = Glide))
        delay(320)
        away.animateTo(1f, tween(480, easing = Smooth))
        done()
    }

    Canvas(Modifier.fillMaxSize()) {
        val w = size.width; val h = size.height
        val a = current
        drawRect(Color.Black.copy(alpha = 1f - away.value))

        // 1. The photo, fading out as the page lifts.
        if (lift.value < 1f) {
            drawImage(
                a.photo,
                dstOffset = IntOffset.Zero,
                dstSize = IntSize(w.roundToInt(), h.roundToInt()),
                alpha = (1f - lift.value) * (1f - away.value),
            )
        }
        val frame = listOf(Offset(0f, 0f), Offset(w, 0f), Offset(w, h), Offset(0f, h))
        val q = a.quad
        if (q == null) {
            // Still checking the photo: a light band sweeps down the frozen picture.
            val y = h * sweep
            val band = 90.dp.toPx()
            drawRect(
                Brush.verticalGradient(0f to Color.Transparent, 1f to Accent.copy(alpha = 0.22f), startY = y - band, endY = y),
                topLeft = Offset(0f, y - band), size = Size(w, band),
            )
            drawLine(Accent.copy(alpha = 0.8f), Offset(0f, y), Offset(w, y), strokeWidth = 2.dp.toPx())
            return@Canvas
        }
        val corners = q.points.mapIndexed { i, p -> lerp(frame[i], Offset(p.x * w, p.y * h), outline.value) }
        if (lift.value < 1f) {
            val path = Path().apply {
                moveTo(corners[0].x, corners[0].y)
                for (k in 1 until 4) lineTo(corners[k].x, corners[k].y)
                close()
            }
            val alpha = 1f - lift.value
            drawPath(path, Accent.copy(alpha = 0.22f * alpha))
            drawPath(path, Accent.copy(alpha = alpha), style = Stroke(width = 3.dp.toPx()))
            corners.forEach { drawCircle(Color.White.copy(alpha = alpha), 6.dp.toPx(), it) }
        }

        // 2-4. The flat page card.
        val flat = a.flat ?: return@Canvas
        if (lift.value <= 0f) return@Canvas
        val start = Rect(
            corners.minOf { it.x }, corners.minOf { it.y },
            corners.maxOf { it.x }, corners.maxOf { it.y },
        )
        val margin = 20.dp.toPx()
        val scale = min((w - 2 * margin) / flat.width, (h - 2 * margin) / flat.height)
        val cw = flat.width * scale; val ch = flat.height * scale
        val centred = Rect(Offset((w - cw) / 2, (h - ch) / 2), Size(cw, ch))
        val small = Rect(Offset(w - cw * 0.18f - margin, h - ch * 0.18f - margin), Size(cw * 0.18f, ch * 0.18f))
        val card = lerp(lerp(start, centred, lift.value), small, away.value)
        val fade = 1f - away.value * 0.6f

        drawRect(Color.Black.copy(alpha = 0.35f * fade), topLeft = card.topLeft + Offset(0f, 6.dp.toPx()), size = card.size)
        drawPage(flat, card, fade)
        val enhanced = a.enhanced
        if (enhanced != null && scan.value > 0f) {
            val cut = card.top + card.height * scan.value
            clipRect(card.left, card.top, card.right, cut) { drawPage(enhanced, card, fade) }
            if (scan.value < 1f) {
                val glow = 26.dp.toPx()
                drawRect(
                    Brush.verticalGradient(
                        0f to Color.Transparent, 0.75f to Accent.copy(alpha = 0.45f), 1f to Color.Transparent,
                        startY = cut - glow, endY = cut + glow * 0.3f,
                    ),
                    topLeft = Offset(card.left, cut - glow),
                    size = Size(card.width, glow * 1.3f),
                )
                drawLine(Color.White, Offset(card.left, cut), Offset(card.right, cut), strokeWidth = 2.dp.toPx())
            }
        }
    }
}

private fun DrawScope.drawPage(img: ImageBitmap, r: Rect, alpha: Float) {
    drawImage(
        img,
        srcOffset = IntOffset.Zero,
        srcSize = IntSize(img.width, img.height),
        dstOffset = IntOffset(r.left.roundToInt(), r.top.roundToInt()),
        dstSize = IntSize(r.width.roundToInt().coerceAtLeast(1), r.height.roundToInt().coerceAtLeast(1)),
        alpha = alpha,
    )
}
