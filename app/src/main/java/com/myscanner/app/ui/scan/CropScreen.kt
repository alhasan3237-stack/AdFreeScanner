package com.myscanner.app.ui.scan

import android.graphics.Bitmap
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.runtime.mutableStateMapOf
import androidx.compose.ui.input.pointer.positionChange
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.AutoFixHigh
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.CropFree
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.ClipOp
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathFillType
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipPath
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.myscanner.app.scan.ImageProcessor
import com.myscanner.app.scan.P
import com.myscanner.app.scan.PreviewRenderer
import com.myscanner.app.scan.Quad
import com.myscanner.app.scan.ScanPage
import com.myscanner.app.scan.ScanSession
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import kotlin.math.min
import kotlin.math.roundToInt

private val Handle = Color(0xFF2EC4B6)

/**
 * Fix the page corners by hand. Drag a corner, or drag the middle of a side to move the whole side.
 * A magnifier shows exactly where the corner sits while you drag. Swipe (away from the handles)
 * to move to the other pages; Done keeps the changes on every page you adjusted.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun CropScreen(session: ScanSession, pageId: Long, onDone: () -> Unit) {
    val pages by session.pages.collectAsStateWithLifecycle()
    if (pages.isEmpty()) {
        LaunchedEffect(Unit) { onDone() }
        return
    }
    val startIndex = remember { session.pages.value.indexOfFirst { it.id == pageId }.coerceAtLeast(0) }
    val pagerState = rememberPagerState(initialPage = startIndex) { pages.size }
    val scope = rememberCoroutineScope()
    /** Corners changed on this screen, per page (kept until Done or Cancel). */
    val edits = remember { mutableStateMapOf<Long, Quad>() }
    var detecting by remember { mutableStateOf(false) }
    var activeHandle by remember { mutableStateOf<Int?>(null) }

    val current = pages.getOrNull(pagerState.currentPage.coerceIn(0, pages.size - 1)) ?: pages[0]
    fun quadOf(page: ScanPage): Quad = edits[page.id] ?: page.quad
    val currentQuad = quadOf(current)
    val allValid = edits.values.all { it.isConvex() }

    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(if (pages.size > 1) "Corners · page ${pagerState.currentPage + 1} of ${pages.size}" else "Adjust corners")
                },
                navigationIcon = { IconButton(onClick = onDone) { Icon(Icons.Outlined.Close, "Cancel") } },
                actions = {
                    Button(
                        onClick = {
                            edits.forEach { (id, q) -> session.update(id) { it.copy(quad = q, detected = true) } }
                            onDone()
                        },
                        enabled = allValid,
                        modifier = Modifier.padding(end = 8.dp),
                    ) { Text("Done") }
                },
            )
        },
        bottomBar = {
            Surface(tonalElevation = 3.dp) {
                Column(Modifier.fillMaxWidth().navigationBarsPadding().padding(vertical = 8.dp)) {
                    Text(
                        when {
                            !currentQuad.isConvex() -> "The corners are crossed – drag them apart"
                            pages.size > 1 -> "Drag the corners or sides · swipe for other pages"
                            else -> "Drag the corners or sides onto the page's edges"
                        },
                        style = MaterialTheme.typography.bodySmall,
                        color = if (currentQuad.isConvex()) MaterialTheme.colorScheme.onSurfaceVariant else MaterialTheme.colorScheme.error,
                        modifier = Modifier.align(Alignment.CenterHorizontally).padding(bottom = 4.dp),
                    )
                    Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                        TextButton(
                            onClick = {
                                detecting = true
                                val page = current
                                scope.launch {
                                    val (q, _) = withContext(PreviewRenderer.dispatcher) {
                                        runCatching { ImageProcessor.detect(page.file) }.getOrElse { Quad.FULL to false }
                                    }
                                    edits[page.id] = q
                                    detecting = false
                                }
                            },
                            enabled = !detecting,
                        ) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Icon(Icons.Outlined.AutoFixHigh, null)
                                Text("Auto detect", style = MaterialTheme.typography.labelMedium)
                            }
                        }
                        TextButton(onClick = { edits[current.id] = Quad(P(0f, 0f), P(1f, 0f), P(1f, 1f), P(0f, 1f)) }) {
                            Column(horizontalAlignment = Alignment.CenterHorizontally) {
                                Icon(Icons.Outlined.CropFree, null)
                                Text("Whole photo", style = MaterialTheme.typography.labelMedium)
                            }
                        }
                    }
                }
            }
        },
    ) { padding ->
        HorizontalPager(
            state = pagerState,
            userScrollEnabled = activeHandle == null && !detecting,
            key = { i -> pages.getOrNull(i)?.id ?: i },
            modifier = Modifier.padding(padding).fillMaxSize().background(Color(0xFF15181B)),
        ) { index ->
            val page = pages.getOrNull(index) ?: return@HorizontalPager
            val isCurrent = index == pagerState.currentPage
            CropPage(
                page = page,
                quad = quadOf(page),
                busy = detecting && isCurrent,
                activeHandle = if (isCurrent) activeHandle else null,
                onActiveHandle = { activeHandle = it },
                onQuad = { edits[page.id] = it },
            )
        }
    }
}

/** One page of the corner screen: the photo with its draggable outline. */
@Composable
private fun CropPage(
    page: ScanPage,
    quad: Quad,
    busy: Boolean,
    activeHandle: Int?,
    onActiveHandle: (Int?) -> Unit,
    onQuad: (Quad) -> Unit,
) {
    var image by remember(page.file) { mutableStateOf<ImageBitmap?>(null) }
    LaunchedEffect(page.file) {
        image = withContext(PreviewRenderer.dispatcher) {
            runCatching {
                val mat = ImageProcessor.load(page.file, 1600)
                val bmp: Bitmap = ImageProcessor.toBitmap(mat)
                mat.release()
                bmp.asImageBitmap()
            }.getOrNull()
        }
    }
    Box(Modifier.fillMaxSize(), contentAlignment = Alignment.Center) {
        val img = image
        if (img == null || busy) CircularProgressIndicator()
        if (img != null) {
            CropCanvas(
                image = img,
                quad = quad,
                activeHandle = activeHandle,
                onActiveHandle = onActiveHandle,
                onQuad = onQuad,
            )
        }
    }
}

/** Where the photo sits inside the canvas. */
private data class Fit(val left: Float, val top: Float, val width: Float, val height: Float) {
    fun toScreen(p: P) = Offset(left + p.x * width, top + p.y * height)
    fun toImage(o: Offset) = P(((o.x - left) / width).coerceIn(0f, 1f), ((o.y - top) / height).coerceIn(0f, 1f))
}

private fun fitImage(canvas: Size, imgW: Int, imgH: Int, margin: Float): Fit {
    val availW = canvas.width - margin * 2
    val availH = canvas.height - margin * 2
    val scale = min(availW / imgW, availH / imgH)
    val w = imgW * scale
    val h = imgH * scale
    return Fit((canvas.width - w) / 2, (canvas.height - h) / 2, w, h)
}

/** Handle positions: 0–3 corners (TL, TR, BR, BL), 4–7 middle of each side. */
private fun handlePoints(q: Quad): List<P> {
    val c = q.points
    return c + (0 until 4).map { i ->
        val a = c[i]
        val b = c[(i + 1) % 4]
        P((a.x + b.x) / 2, (a.y + b.y) / 2)
    }
}

@Composable
private fun CropCanvas(
    image: ImageBitmap,
    quad: Quad,
    activeHandle: Int?,
    onActiveHandle: (Int?) -> Unit,
    onQuad: (Quad) -> Unit,
) {
    // The gesture code reads the latest values through these holders.
    val quadHolder = remember { mutableStateOf(quad) }
    quadHolder.value = quad

    Canvas(
        Modifier
            .fillMaxSize()
            .pointerInput(image) {
                val margin = 28.dp.toPx()
                val grab = 40.dp.toPx()
                awaitEachGesture {
                    val down = awaitFirstDown(requireUnconsumed = false)
                    val fit = fitImage(Size(size.width.toFloat(), size.height.toFloat()), image.width, image.height, margin)
                    val handles = handlePoints(quadHolder.value).map { fit.toScreen(it) }
                    val nearest = handles.indices.minBy { (handles[it] - down.position).getDistance() }
                    // Not on a handle: leave the touch alone, so a swipe moves to the next page.
                    if ((handles[nearest] - down.position).getDistance() > grab) return@awaitEachGesture
                    down.consume()
                    val grabOffset = handles[nearest] - down.position
                    var finger = down.position
                    onActiveHandle(nearest)
                    try {
                        while (true) {
                            val event = awaitPointerEvent()
                            val change = event.changes.firstOrNull { it.id == down.id } ?: break
                            if (!change.pressed) break
                            val drag = change.positionChange()
                            if (drag != Offset.Zero) {
                                change.consume()
                                finger += drag
                                val f = fitImage(Size(size.width.toFloat(), size.height.toFloat()), image.width, image.height, margin)
                                val newQuad = moveHandle(quadHolder.value, nearest, f, finger + grabOffset, drag)
                                quadHolder.value = newQuad
                                onQuad(newQuad)
                            }
                        }
                    } finally {
                        onActiveHandle(null)
                    }
                }
            },
    ) {
        val fit = fitImage(size, image.width, image.height, 28.dp.toPx())
        drawImage(
            image,
            srcOffset = IntOffset.Zero,
            srcSize = IntSize(image.width, image.height),
            dstOffset = IntOffset(fit.left.roundToInt(), fit.top.roundToInt()),
            dstSize = IntSize(fit.width.roundToInt(), fit.height.roundToInt()),
        )

        val corners = quad.points.map { fit.toScreen(it) }
        val outline = Path().apply {
            moveTo(corners[0].x, corners[0].y)
            for (i in 1 until 4) lineTo(corners[i].x, corners[i].y)
            close()
        }
        // Darken everything outside the page.
        val shade = Path().apply {
            fillType = PathFillType.EvenOdd
            addRect(Rect(fit.left, fit.top, fit.left + fit.width, fit.top + fit.height))
            addPath(outline)
        }
        drawPath(shade, Color.Black.copy(alpha = 0.5f))
        drawPath(outline, Handle, style = Stroke(width = 2.5.dp.toPx()))

        handlePoints(quad).forEachIndexed { i, p ->
            val c = fit.toScreen(p)
            val active = i == activeHandle
            if (i < 4) {
                drawCircle(Color.White, radius = (if (active) 14 else 11).dp.toPx(), center = c)
                drawCircle(Handle, radius = (if (active) 10 else 8).dp.toPx(), center = c)
            } else {
                drawCircle(Color.White, radius = 7.dp.toPx(), center = c)
                drawCircle(Handle, radius = 5.dp.toPx(), center = c)
            }
        }

        activeHandle?.let { h -> drawMagnifier(image, fit, handlePoints(quad)[h], corners) }
    }
}

private fun moveHandle(q: Quad, handle: Int, fit: Fit, target: Offset, drag: Offset): Quad {
    return if (handle < 4) {
        q.with(handle, fit.toImage(target))
    } else {
        // Side handle: move both of its corners by the same amount.
        val i = handle - 4
        val j = (i + 1) % 4
        val dx = drag.x / fit.width
        val dy = drag.y / fit.height
        val a = q.points[i]
        val b = q.points[j]
        q.with(i, P((a.x + dx).coerceIn(0f, 1f), (a.y + dy).coerceIn(0f, 1f)))
            .with(j, P((b.x + dx).coerceIn(0f, 1f), (b.y + dy).coerceIn(0f, 1f)))
    }
}

/** A round zoomed view of the photo around the handle, shown at the top corner away from the finger. */
private fun DrawScope.drawMagnifier(image: ImageBitmap, fit: Fit, at: P, corners: List<Offset>) {
    val radius = 60.dp.toPx()
    val zoom = 2.5f
    val handleOnScreen = fit.toScreen(at)
    val centerX = if (handleOnScreen.x < size.width / 2) size.width - radius - 16.dp.toPx() else radius + 16.dp.toPx()
    val center = Offset(centerX, radius + 16.dp.toPx())

    val pxPerImage = fit.width / image.width * zoom // screen px per image px inside the loupe
    val srcHalf = radius / pxPerImage
    val srcCx = at.x * image.width
    val srcCy = at.y * image.height
    val srcLeft = (srcCx - srcHalf).roundToInt()
    val srcTop = (srcCy - srcHalf).roundToInt()
    val srcSize = (srcHalf * 2).roundToInt().coerceAtLeast(1)

    val circle = Path().apply { addOval(Rect(center, radius)) }
    clipPath(circle, clipOp = ClipOp.Intersect) {
        drawRect(Color.Black, topLeft = Offset(center.x - radius, center.y - radius), size = Size(radius * 2, radius * 2))
        // Clamp the source to the photo; shift the destination to match.
        val sx = srcLeft.coerceIn(0, image.width - 1)
        val sy = srcTop.coerceIn(0, image.height - 1)
        val sw = min(srcSize - (sx - srcLeft), image.width - sx).coerceAtLeast(1)
        val sh = min(srcSize - (sy - srcTop), image.height - sy).coerceAtLeast(1)
        val dx = center.x - radius + (sx - srcLeft) * pxPerImage
        val dy = center.y - radius + (sy - srcTop) * pxPerImage
        drawImage(
            image,
            srcOffset = IntOffset(sx, sy),
            srcSize = IntSize(sw, sh),
            dstOffset = IntOffset(dx.roundToInt(), dy.roundToInt()),
            dstSize = IntSize((sw * pxPerImage).roundToInt(), (sh * pxPerImage).roundToInt()),
        )
        // The page outline, zoomed the same way.
        val toLoupe = { o: Offset ->
            val imgX = (o.x - fit.left) / fit.width * image.width
            val imgY = (o.y - fit.top) / fit.height * image.height
            Offset(center.x + (imgX - srcCx) * pxPerImage, center.y + (imgY - srcCy) * pxPerImage)
        }
        val zoomed = corners.map(toLoupe)
        val path = Path().apply {
            moveTo(zoomed[0].x, zoomed[0].y)
            for (k in 1 until 4) lineTo(zoomed[k].x, zoomed[k].y)
            close()
        }
        drawPath(path, Handle, style = Stroke(width = 2.dp.toPx()))
        // Crosshair
        val arm = 10.dp.toPx()
        drawLine(Color.White, Offset(center.x - arm, center.y), Offset(center.x + arm, center.y), strokeWidth = 1.5.dp.toPx())
        drawLine(Color.White, Offset(center.x, center.y - arm), Offset(center.x, center.y + arm), strokeWidth = 1.5.dp.toPx())
    }
    drawCircle(Color.White, radius = radius, center = center, style = Stroke(width = 3.dp.toPx()))
}
