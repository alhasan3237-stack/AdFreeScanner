package com.myscanner.app.ui.components

import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.gestures.calculateCentroid
import androidx.compose.foundation.gestures.calculatePan
import androidx.compose.foundation.gestures.calculateZoom
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clipToBounds
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChanged
import androidx.compose.ui.layout.onSizeChanged
import androidx.compose.ui.unit.IntSize
import kotlin.math.max

private const val MAX_ZOOM = 6f
private const val DOUBLE_TAP_ZOOM = 2.5f

/**
 * Pinch to zoom, drag to move around, double-tap to zoom in / back out.
 * One-finger swipes pass through when not zoomed, so a page pager still swipes.
 * [onZoom] reports the current zoom (1 = not zoomed).
 */
@Composable
fun ZoomableBox(
    modifier: Modifier = Modifier,
    resetKey: Any? = null,
    onZoom: (Float) -> Unit = {},
    content: @Composable () -> Unit,
) {
    var scale by remember { mutableFloatStateOf(1f) }
    var offset by remember { mutableStateOf(Offset.Zero) }
    var size by remember { mutableStateOf(IntSize.Zero) }
    val currentOnZoom by rememberUpdatedState(onZoom)

    LaunchedEffect(resetKey) { scale = 1f; offset = Offset.Zero; currentOnZoom(1f) }

    fun clamp(o: Offset, s: Float): Offset {
        val maxX = max(0f, size.width * (s - 1) / 2)
        val maxY = max(0f, size.height * (s - 1) / 2)
        return Offset(o.x.coerceIn(-maxX, maxX), o.y.coerceIn(-maxY, maxY))
    }

    fun zoomTo(newScale: Float, focus: Offset) {
        val s = newScale.coerceIn(1f, MAX_ZOOM)
        val center = Offset(size.width / 2f, size.height / 2f)
        // Keep the point under the fingers in place while zooming.
        val o = (focus - center) - (focus - center - offset) * (s / scale)
        scale = s
        offset = clamp(o, s)
        currentOnZoom(s)
    }

    Box(
        modifier
            .clipToBounds()
            .onSizeChanged { size = it }
            .pointerInput(Unit) {
                detectTapGestures(
                    onDoubleTap = { tap ->
                        if (scale > 1.05f) { scale = 1f; offset = Offset.Zero; currentOnZoom(1f) }
                        else zoomTo(DOUBLE_TAP_ZOOM, tap)
                    },
                )
            }
            .pointerInput(Unit) {
                awaitEachGesture {
                    awaitFirstDown(requireUnconsumed = false)
                    do {
                        val event = awaitPointerEvent()
                        val fingers = event.changes.count { it.pressed }
                        if (fingers >= 2 || scale > 1.01f) {
                            val zoom = event.calculateZoom()
                            val pan = event.calculatePan()
                            val centroid = event.calculateCentroid(useCurrent = false)
                            if (zoom != 1f && centroid != Offset.Unspecified) zoomTo(scale * zoom, centroid)
                            offset = clamp(offset + pan, scale)
                            event.changes.forEach { if (it.positionChanged()) it.consume() }
                        }
                    } while (event.changes.any { it.pressed })
                }
            },
        contentAlignment = Alignment.Center,
    ) {
        Box(
            Modifier.fillMaxSize().graphicsLayer {
                scaleX = scale
                scaleY = scale
                translationX = offset.x
                translationY = offset.y
            },
            contentAlignment = Alignment.Center,
        ) { content() }
    }
}
