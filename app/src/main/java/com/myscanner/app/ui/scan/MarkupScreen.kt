package com.myscanner.app.ui.scan

import android.content.Context
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.awaitEachGesture
import androidx.compose.foundation.gestures.awaitFirstDown
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.Undo
import androidx.compose.material.icons.outlined.BorderColor
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.Gesture
import androidx.compose.material.icons.outlined.TextFields
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Slider
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateListOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberUpdatedState
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.BlendMode
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.ImageBitmap
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.DrawScope
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.drawIntoCanvas
import androidx.compose.ui.graphics.nativeCanvas
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.input.pointer.positionChange
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.IntOffset
import androidx.compose.ui.unit.IntSize
import androidx.compose.ui.unit.dp
import com.myscanner.app.scan.Mark
import com.myscanner.app.scan.Marks
import com.myscanner.app.scan.P
import com.myscanner.app.scan.PreviewRenderer
import com.myscanner.app.scan.ScanSession
import com.myscanner.app.ui.components.ConfirmDialog
import kotlinx.coroutines.withContext
import java.io.File
import kotlin.math.max
import kotlin.math.min
import kotlin.math.roundToInt

private enum class DrawTool { PEN, HIGHLIGHT }

private val PenColors = listOf(0xFF111111, 0xFF1E4FD8, 0xFFD32F2F, 0xFF1B8E3E).map { it.toInt() }
private val MarkerColors = listOf(0xFFFFF176, 0xFFA5F3A1, 0xFFFFB0CF, 0xFF9BE7FF).map { it.toInt() }
/** Line thickness as a fraction of the page's shorter side: thin, medium, thick. */
private val PenWidths = listOf(0.003f, 0.006f, 0.011f)
private val MarkerWidths = listOf(0.02f, 0.035f, 0.055f)

/**
 * Typed text or the signature, chosen but not yet fixed on the page: it floats over the page and
 * can be dragged and resized, then "Place" fixes it.
 */
private data class Floating(
    val text: String? = null,
    /** Signature strokes inside a 1 x 1 box. */
    val strokes: List<Mark.Stroke>? = null,
    /** Signature: width / height. */
    val aspect: Float = 1f,
    val cx: Float = 0.5f,
    val cy: Float = 0.5f,
    /** Text: letter height as a fraction of the page's shorter side. Signature: width as a fraction of the page width. */
    val size: Float,
)

/** Where the page picture sits inside the drawing area. */
private data class PageFit(val left: Float, val top: Float, val width: Float, val height: Float) {
    val shorter: Float get() = min(width, height)
    fun toScreen(p: P) = Offset(left + p.x * width, top + p.y * height)
    fun toPage(o: Offset) = P(((o.x - left) / width).coerceIn(0f, 1f), ((o.y - top) / height).coerceIn(0f, 1f))
    fun contains(o: Offset, slack: Float) =
        o.x >= left - slack && o.x <= left + width + slack && o.y >= top - slack && o.y <= top + height + slack
}

private fun pageFit(canvas: Size, imgW: Int, imgH: Int, margin: Float): PageFit {
    val scale = min((canvas.width - margin * 2) / imgW, (canvas.height - margin * 2) / imgH)
    val w = imgW * scale
    val h = imgH * scale
    return PageFit((canvas.width - w) / 2, (canvas.height - h) / 2, w, h)
}

/** The user's signature, drawn once and kept on the phone for next time. */
private object SignatureStore {
    private fun file(context: Context) = File(context.filesDir, "signature.txt")

    fun load(context: Context): Pair<List<Mark.Stroke>, Float>? = runCatching {
        val lines = file(context).readLines()
        val aspect = lines.first().removePrefix("A|").toFloat()
        val strokes = Marks.decode(lines.drop(1).joinToString("\n")).filterIsInstance<Mark.Stroke>()
        if (strokes.isEmpty() || aspect <= 0f) null else strokes to aspect
    }.getOrNull()

    /** [strokes] in screen pixels as drawn on the pad; [penPx] = the pad's pen thickness. */
    fun save(context: Context, strokes: List<List<Offset>>, penPx: Float): Pair<List<Mark.Stroke>, Float>? {
        val all = strokes.flatten()
        if (all.size < 2) return null
        val minX = all.minOf { it.x }; val minY = all.minOf { it.y }
        val bw = max(all.maxOf { it.x } - minX, 1f); val bh = max(all.maxOf { it.y } - minY, 1f)
        val norm = strokes.filter { it.isNotEmpty() }.map { st ->
            Mark.Stroke(st.map { P((it.x - minX) / bw, (it.y - minY) / bh) }, 0xFF000000.toInt(), penPx / bw, false)
        }
        val aspect = (bw / bh).coerceIn(0.2f, 20f)
        runCatching { file(context).writeText("A|$aspect\n" + Marks.encode(norm)) }
        return norm to aspect
    }
}

/** The marks a floating item becomes when it is placed. [pageAspect] = page width / height. */
private fun Floating.toMarks(color: Int, pageAspect: Float): List<Mark> {
    text?.let { return listOf(Mark.Label(it, cx, cy, size, color)) }
    val st = strokes ?: return emptyList()
    val h = size * pageAspect / aspect // the signature's height as a fraction of the page height
    return st.map { s ->
        Mark.Stroke(
            points = s.points.map { P((cx + (it.x - 0.5f) * size).coerceIn(0f, 1f), (cy + (it.y - 0.5f) * h).coerceIn(0f, 1f)) },
            color = color,
            width = s.width * size * max(1f, pageAspect),
            highlighter = false,
        )
    }
}

/**
 * Draw on a page: pen, highlighter, typed text and a saved signature. Everything stays editable
 * until the scan is saved (and afterwards too, when original photos are kept).
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun MarkupScreen(session: ScanSession, pageId: Long, onDone: () -> Unit) {
    val context = LocalContext.current
    val page = remember { session.page(pageId) }
    if (page == null) {
        LaunchedEffect(Unit) { onDone() }
        return
    }
    var image by remember { mutableStateOf<ImageBitmap?>(null) }
    LaunchedEffect(Unit) {
        image = withContext(PreviewRenderer.dispatcher) {
            runCatching { PreviewRenderer.render(page, PreviewRenderer.previewSide, withMarks = false) }.getOrNull()
        }?.asImageBitmap()
    }

    val marks = remember { mutableStateListOf<Mark>().apply { addAll(page.marks) } }
    /** How many marks each step added, so Undo removes a whole signature at once. */
    val steps = remember { mutableStateListOf<Int>().apply { repeat(page.marks.size) { add(1) } } }
    var changed by remember { mutableStateOf(false) }
    var tool by remember { mutableStateOf(DrawTool.PEN) }
    var penColor by remember { mutableIntStateOf(PenColors[1]) }
    var markerColor by remember { mutableIntStateOf(MarkerColors[0]) }
    var widthIndex by remember { mutableIntStateOf(1) }
    var floating by remember { mutableStateOf<Floating?>(null) }
    var askText by remember { mutableStateOf(false) }
    var askSignature by remember { mutableStateOf(false) }
    var confirmLeave by remember { mutableStateOf(false) }
    /** The line being drawn right now, in screen pixels. */
    val live = remember { mutableStateListOf<Offset>() }

    fun leave() { if (changed || floating != null) confirmLeave = true else onDone() }
    BackHandler { leave() }

    fun add(new: List<Mark>) {
        if (new.isEmpty()) return
        marks.addAll(new); steps.add(new.size); changed = true
    }
    fun undo() {
        val n = steps.removeLastOrNull() ?: return
        repeat(n) { marks.removeLastOrNull() }
        changed = true
    }
    fun place() {
        val f = floating ?: return
        val img = image ?: return
        add(f.toMarks(penColor, img.width.toFloat() / img.height))
        floating = null
    }
    fun startSignature() {
        val saved = SignatureStore.load(context)
        if (saved == null) askSignature = true
        else floating = Floating(strokes = saved.first, aspect = saved.second, size = 0.36f, cy = 0.8f)
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Draw / sign") },
                navigationIcon = { IconButton(onClick = { leave() }) { Icon(Icons.Outlined.Close, "Cancel") } },
                actions = {
                    IconButton(onClick = { undo() }, enabled = steps.isNotEmpty() && floating == null) {
                        Icon(Icons.AutoMirrored.Outlined.Undo, "Undo")
                    }
                    Button(
                        onClick = {
                            place()
                            session.update(pageId) { it.copy(marks = marks.toList()) }
                            onDone()
                        },
                        modifier = Modifier.padding(end = 8.dp),
                    ) { Text("Done") }
                },
            )
        },
        bottomBar = {
            Surface(tonalElevation = 3.dp) {
                Column(Modifier.fillMaxWidth().navigationBarsPadding().padding(horizontal = 12.dp, vertical = 8.dp)) {
                    val f = floating
                    if (f != null) {
                        Text(
                            "Drag it into place, set the size, then tap Place",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                            modifier = Modifier.align(Alignment.CenterHorizontally),
                        )
                        val range = if (f.text != null) 0.015f..0.12f else 0.12f..0.9f
                        Slider(value = f.size, onValueChange = { floating = f.copy(size = it) }, valueRange = range)
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.spacedBy(8.dp), verticalAlignment = Alignment.CenterVertically) {
                            TextButton(onClick = { floating = null }) { Text("Cancel") }
                            if (f.strokes != null) TextButton(onClick = { floating = null; askSignature = true }) { Text("New signature") }
                            Spacer(Modifier.weight(1f))
                            ColorDots(PenColors, penColor) { penColor = it }
                            Button(onClick = { place() }) { Text("Place") }
                        }
                    } else {
                        Row(Modifier.fillMaxWidth(), verticalAlignment = Alignment.CenterVertically) {
                            if (tool == DrawTool.PEN) ColorDots(PenColors, penColor) { penColor = it }
                            else ColorDots(MarkerColors, markerColor) { markerColor = it }
                            Spacer(Modifier.weight(1f))
                            // Thin / medium / thick
                            (0..2).forEach { i ->
                                Box(
                                    Modifier.size(40.dp).clip(CircleShape).clickable(onClickLabel = "Thickness ${i + 1}") { widthIndex = i },
                                    contentAlignment = Alignment.Center,
                                ) {
                                    Box(
                                        Modifier.size((6 + i * 6).dp).clip(CircleShape).background(
                                            if (widthIndex == i) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outline
                                        )
                                    )
                                }
                            }
                        }
                        Row(Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceEvenly) {
                            ToolButton("Pen", tool == DrawTool.PEN, { Icon(Icons.Outlined.Edit, null) }) { tool = DrawTool.PEN }
                            ToolButton("Highlighter", tool == DrawTool.HIGHLIGHT, { Icon(Icons.Outlined.BorderColor, null) }) { tool = DrawTool.HIGHLIGHT }
                            ToolButton("Text", false, { Icon(Icons.Outlined.TextFields, null) }) { askText = true }
                            ToolButton("Signature", false, { Icon(Icons.Outlined.Gesture, null) }) { startSignature() }
                        }
                    }
                }
            }
        },
    ) { padding ->
        Box(Modifier.padding(padding).fillMaxSize().background(Color(0xFF15181B)), contentAlignment = Alignment.Center) {
            val img = image
            if (img == null) {
                CircularProgressIndicator()
            } else {
                val floatingNow by rememberUpdatedState(floating)
                val toolNow by rememberUpdatedState(tool)
                val colorNow by rememberUpdatedState(if (tool == DrawTool.PEN) penColor else markerColor)
                val widthNow by rememberUpdatedState((if (tool == DrawTool.PEN) PenWidths else MarkerWidths)[widthIndex])
                val textPaint = remember { android.graphics.Paint(android.graphics.Paint.ANTI_ALIAS_FLAG) }
                Canvas(
                    Modifier.fillMaxSize().pointerInput(img) {
                        val margin = 12.dp.toPx()
                        awaitEachGesture {
                            val down = awaitFirstDown(requireUnconsumed = false)
                            val fit = pageFit(Size(size.width.toFloat(), size.height.toFloat()), img.width, img.height, margin)
                            val moving = floatingNow != null
                            if (!moving && !fit.contains(down.position, 12.dp.toPx())) return@awaitEachGesture
                            down.consume()
                            if (!moving) { live.clear(); live.add(down.position) }
                            while (true) {
                                val event = awaitPointerEvent()
                                val change = event.changes.firstOrNull { it.id == down.id } ?: break
                                if (!change.pressed) break
                                val drag = change.positionChange()
                                if (drag == Offset.Zero) continue
                                change.consume()
                                val f = floatingNow
                                if (moving) {
                                    if (f != null) floating = f.copy(
                                        cx = (f.cx + drag.x / fit.width).coerceIn(0f, 1f),
                                        cy = (f.cy + drag.y / fit.height).coerceIn(0f, 1f),
                                    )
                                } else if ((change.position - live.last()).getDistance() > 1.5f) {
                                    live.add(change.position)
                                }
                            }
                            if (!moving && live.isNotEmpty()) {
                                add(listOf(Mark.Stroke(live.map { fit.toPage(it) }, colorNow, widthNow, toolNow == DrawTool.HIGHLIGHT)))
                                live.clear()
                            }
                        }
                    },
                ) {
                    val fit = pageFit(size, img.width, img.height, 12.dp.toPx())
                    drawImage(
                        img,
                        srcOffset = IntOffset.Zero, srcSize = IntSize(img.width, img.height),
                        dstOffset = IntOffset(fit.left.roundToInt(), fit.top.roundToInt()),
                        dstSize = IntSize(fit.width.roundToInt(), fit.height.roundToInt()),
                    )
                    for (m in marks) drawMark(m, fit, textPaint)
                    if (live.isNotEmpty()) {
                        drawInk(live, Color(colorNow), widthNow * fit.shorter, toolNow == DrawTool.HIGHLIGHT)
                    }
                    floating?.let { f ->
                        val pageAspect = img.width.toFloat() / img.height
                        val centre = fit.toScreen(P(f.cx, f.cy))
                        val box: Size
                        if (f.text != null) {
                            box = drawLabel(f.text, centre, f.size * fit.shorter, penColor, 0, textPaint)
                        } else {
                            f.toMarks(penColor, pageAspect).forEach { drawMark(it, fit, textPaint) }
                            box = Size(f.size * fit.width, f.size * pageAspect / f.aspect * fit.height)
                        }
                        val pad = 8.dp.toPx()
                        drawRect(
                            Color(0xFF2EC4B6),
                            topLeft = Offset(centre.x - box.width / 2 - pad, centre.y - box.height / 2 - pad),
                            size = Size(box.width + pad * 2, box.height + pad * 2),
                            style = Stroke(width = 1.5.dp.toPx(), pathEffect = PathEffect.dashPathEffect(floatArrayOf(14f, 10f))),
                        )
                    }
                }
            }
        }
    }

    if (askText) {
        var text by remember { mutableStateOf("") }
        AlertDialog(
            onDismissRequest = { askText = false },
            title = { Text("Add text") },
            text = {
                OutlinedTextField(
                    value = text, onValueChange = { text = it }, maxLines = 4,
                    placeholder = { Text("Name, date, a note…") },
                )
            },
            confirmButton = {
                TextButton(
                    onClick = { askText = false; if (text.isNotBlank()) floating = Floating(text = text.trim(), size = 0.04f) },
                    enabled = text.isNotBlank(),
                ) { Text("Add") }
            },
            dismissButton = { TextButton(onClick = { askText = false }) { Text("Cancel") } },
        )
    }
    if (askSignature) {
        SignaturePad(
            onUse = { strokes, penPx ->
                askSignature = false
                SignatureStore.save(context, strokes, penPx)?.let { (st, aspect) ->
                    floating = Floating(strokes = st, aspect = aspect, size = 0.36f, cy = 0.8f)
                }
            },
            onDismiss = { askSignature = false },
        )
    }
    if (confirmLeave) {
        ConfirmDialog(
            title = "Leave without keeping the drawing?",
            message = "What you drew on this screen will be lost.",
            confirmLabel = "Leave",
            onConfirm = { confirmLeave = false; onDone() },
            onDismiss = { confirmLeave = false },
        )
    }
}

/** A white pad to sign on with a finger. The signature is kept on the phone for next time. */
@Composable
private fun SignaturePad(onUse: (List<List<Offset>>, Float) -> Unit, onDismiss: () -> Unit) {
    val strokes = remember { mutableStateListOf<List<Offset>>() }
    val live = remember { mutableStateListOf<Offset>() }
    var penPx by remember { mutableStateOf(6f) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text("Your signature") },
        text = {
            Column {
                Text("Sign in the box with your finger. It is saved on this phone only.", style = MaterialTheme.typography.bodySmall)
                Spacer(Modifier.height(10.dp))
                Canvas(
                    Modifier
                        .fillMaxWidth()
                        .height(170.dp)
                        .clip(RoundedCornerShape(10.dp))
                        .background(Color.White)
                        .border(1.dp, MaterialTheme.colorScheme.outline, RoundedCornerShape(10.dp))
                        .pointerInput(Unit) {
                            penPx = 3.dp.toPx()
                            awaitEachGesture {
                                val down = awaitFirstDown(requireUnconsumed = false)
                                down.consume()
                                live.clear(); live.add(down.position)
                                while (true) {
                                    val event = awaitPointerEvent()
                                    val change = event.changes.firstOrNull { it.id == down.id } ?: break
                                    if (!change.pressed) break
                                    change.consume()
                                    if ((change.position - live.last()).getDistance() > 1.5f) live.add(change.position)
                                }
                                strokes.add(live.toList())
                                live.clear()
                            }
                        },
                ) {
                    val ink = Color(0xFF111111)
                    strokes.forEach { drawInk(it, ink, 3.dp.toPx(), false) }
                    if (live.isNotEmpty()) drawInk(live, ink, 3.dp.toPx(), false)
                }
                TextButton(onClick = { strokes.clear() }, modifier = Modifier.align(Alignment.End)) { Text("Clear") }
            }
        },
        confirmButton = {
            TextButton(onClick = { onUse(strokes.toList(), penPx) }, enabled = strokes.sumOf { it.size } >= 2) { Text("Use") }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
private fun ColorDots(colors: List<Int>, selected: Int, onPick: (Int) -> Unit) {
    Row(horizontalArrangement = Arrangement.spacedBy(4.dp), verticalAlignment = Alignment.CenterVertically) {
        colors.forEach { c ->
            Box(
                Modifier.size(40.dp).clip(CircleShape).clickable(onClickLabel = "Colour") { onPick(c) },
                contentAlignment = Alignment.Center,
            ) {
                Box(
                    Modifier
                        .size(if (c == selected) 30.dp else 22.dp)
                        .clip(CircleShape)
                        .background(Color(c))
                        .border(if (c == selected) 3.dp else 1.dp, MaterialTheme.colorScheme.onSurface.copy(alpha = if (c == selected) 0.9f else 0.3f), CircleShape)
                )
            }
        }
    }
}

@Composable
private fun ToolButton(label: String, selected: Boolean, icon: @Composable () -> Unit, onClick: () -> Unit) {
    TextButton(onClick = onClick) {
        Column(
            horizontalAlignment = Alignment.CenterHorizontally,
            modifier = Modifier
                .clip(RoundedCornerShape(10.dp))
                .background(if (selected) MaterialTheme.colorScheme.primaryContainer else Color.Transparent)
                .padding(horizontal = 10.dp, vertical = 4.dp),
        ) {
            icon()
            Text(label, style = MaterialTheme.typography.labelMedium)
        }
    }
}

// ---------- drawing the marks on screen, the same way they are drawn into the saved page ----------

private fun DrawScope.drawMark(m: Mark, fit: PageFit, paint: android.graphics.Paint) {
    when (m) {
        is Mark.Stroke -> drawInk(m.points.map { fit.toScreen(it) }, Color(m.color or 0xFF000000.toInt()), m.width * fit.shorter, m.highlighter)
        is Mark.Label -> drawLabel(m.text, fit.toScreen(P(m.x, m.y)), m.size * fit.shorter, m.color, m.angle, paint)
    }
}

private fun DrawScope.drawInk(points: List<Offset>, color: Color, widthPx: Float, highlighter: Boolean) {
    if (points.isEmpty()) return
    // A highlighter only tints (multiply): the text under it stays fully dark.
    val blend = if (highlighter) BlendMode.Multiply else BlendMode.SrcOver
    if (points.size == 1) {
        drawCircle(color, max(1f, widthPx / 2), points[0], blendMode = blend)
        return
    }
    val path = Path().apply {
        moveTo(points[0].x, points[0].y)
        for (i in 1 until points.size) lineTo(points[i].x, points[i].y)
    }
    drawPath(path, color, style = Stroke(width = max(1f, widthPx), cap = StrokeCap.Round, join = StrokeJoin.Round), blendMode = blend)
}

/** Draws typed text centred on [centre]; returns the size of its box. */
private fun DrawScope.drawLabel(text: String, centre: Offset, sizePx: Float, color: Int, angle: Int, paint: android.graphics.Paint): Size {
    paint.color = color or 0xFF000000.toInt()
    paint.textSize = max(8f, sizePx)
    val lines = text.split("\n")
    val fm = paint.fontMetrics
    val lineH = fm.descent - fm.ascent
    val tw = lines.maxOf { paint.measureText(it) } + 5f
    val th = lineH * lines.size + 5f
    drawIntoCanvas { canvas ->
        val native = canvas.nativeCanvas
        native.save()
        native.rotate(angle.toFloat(), centre.x, centre.y)
        lines.forEachIndexed { i, line ->
            native.drawText(line, centre.x - tw / 2 + 2f, centre.y - th / 2 + 2f - fm.ascent + i * lineH, paint)
        }
        native.restore()
    }
    return Size(tw, th)
}
