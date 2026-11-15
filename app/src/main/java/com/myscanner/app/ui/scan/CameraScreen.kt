package com.myscanner.app.ui.scan

import android.Manifest
import android.content.pm.PackageManager
import android.util.Size
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.PickVisualMediaRequest
import androidx.activity.result.contract.ActivityResultContracts
import androidx.camera.core.Camera
import androidx.camera.core.CameraSelector
import androidx.camera.core.FocusMeteringAction
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageCapture
import androidx.camera.core.ImageCaptureException
import androidx.camera.core.Preview
import androidx.camera.core.resolutionselector.AspectRatioStrategy
import androidx.camera.core.resolutionselector.ResolutionSelector
import androidx.camera.core.resolutionselector.ResolutionStrategy
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.gestures.detectTapGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.FlashAuto
import androidx.compose.material.icons.outlined.FlashOff
import androidx.compose.material.icons.outlined.FlashOn
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material3.Badge
import androidx.compose.material3.BadgedBox
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.myscanner.app.data.PhotoQuality
import com.myscanner.app.data.ScanFilter
import kotlinx.coroutines.flow.StateFlow
import androidx.compose.runtime.SideEffect
import com.myscanner.app.scan.EdgeAnalyzer
import com.myscanner.app.scan.ImageProcessor
import com.myscanner.app.scan.PreviewRenderer
import com.myscanner.app.scan.Quad
import com.myscanner.app.scan.ScanSession
import com.myscanner.app.scan.importGalleryImage
import com.myscanner.app.ui.components.ConfirmDialog
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors
import android.os.SystemClock
import kotlinx.coroutines.delay
import kotlin.math.abs
import androidx.compose.material3.TextButton
import java.util.concurrent.TimeUnit
import android.graphics.Bitmap
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.compose.animation.core.LinearOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import kotlinx.coroutines.suspendCancellableCoroutine
import kotlinx.coroutines.withTimeoutOrNull
import kotlin.coroutines.resume
import kotlin.math.sqrt

/** One photo at a time is checked for page edges, so pages always stay in the order taken. */
@OptIn(ExperimentalCoroutinesApi::class)
private val detectDispatcher = Dispatchers.IO.limitedParallelism(1)

private val Teal = Color(0xFF2EC4B6)

@Composable
fun CameraScreen(
    session: ScanSession,
    onDone: () -> Unit,
    onClose: () -> Unit,
    /** Play the auto-enhance animation after each photo. */
    animate: Boolean = true,
    /** Start with auto-capture on (takes the photo by itself when the page is steady). */
    autoCaptureDefault: Boolean = false,
    /** Find the page live and draw its outline on the camera picture (Settings > Live edge detection). */
    liveOutline: Boolean = true,
    /** Adding pages while editing a saved document: leaving just goes back to the pages. */
    editing: Boolean = false,
) {
    val context = LocalContext.current
    val scope = rememberCoroutineScope()
    val pages by session.pages.collectAsStateWithLifecycle()
    val processing by session.processing.collectAsStateWithLifecycle()

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA) == PackageManager.PERMISSION_GRANTED
        )
    }
    val askPermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { hasPermission = it }
    LaunchedEffect(Unit) { if (!hasPermission) askPermission.launch(Manifest.permission.CAMERA) }

    // The live outline changes several times a second. Only the outline and the title read it, so
    // the rest of this screen isn't redrawn each time (that made the camera feel heavy).
    val liveQuad = remember { MutableStateFlow<Quad?>(null) }
    val analyzer = remember { EdgeAnalyzer { liveQuad.value = it } }

    // Parts that crashed on this phone are switched off automatically (see NativeGuard).
    var safeMode by remember { mutableStateOf(com.myscanner.app.scan.NativeGuard.isOff(com.myscanner.app.scan.NativeGuard.LIVE)) }
    var flashMode by remember { mutableStateOf(ImageCapture.FLASH_MODE_OFF) }
    var capturing by remember { mutableStateOf(false) }
    var showDiscard by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }
    LaunchedEffect(Unit) {
        com.myscanner.app.scan.NativeGuard.justDisabled?.let {
            error = "The $it crashed on this phone last time, so it was switched off. Scanning still works."
            com.myscanner.app.scan.NativeGuard.justDisabled = null
        }
    }
    /** What the camera is doing right now, shown at the top. */
    var stage by remember { mutableStateOf<String?>(null) }
    /** The picture frozen at the moment of the shot (instant feedback, no flicker). */
    var frozen by remember { mutableStateOf<Bitmap?>(null) }
    val frozenAlpha = remember { Animatable(1f) }
    /** A page whose photo came out blurry: offer to retake it. */
    var blurryPage by remember { mutableStateOf<Long?>(null) }
    var shotCounter by remember { mutableStateOf(0L) }

    val quality = session.photoQuality
    val highRes = quality == PhotoQuality.HIGH
    val imageCapture = remember(quality) {
        ImageCapture.Builder()
            // Fastest shutter: the photo is taken the moment the phone is still, so it isn't shaken.
            .setCaptureMode(ImageCapture.CAPTURE_MODE_MINIMIZE_LATENCY)
            .setJpegQuality(if (highRes) 95 else 90)
            .setResolutionSelector(
                ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    // The biggest 4:3 size up to the chosen quality (13 MP or 5 MP).
                    .setResolutionStrategy(
                        ResolutionStrategy(
                            Size(quality.captureWidth, quality.captureHeight),
                            ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER,
                        )
                    )
                    .build()
            )
            .build()
    }
    LaunchedEffect(flashMode) { imageCapture.flashMode = flashMode }

    fun leave() {
        if (editing || pages.isEmpty()) onClose() else showDiscard = true
    }
    BackHandler { leave() }

    var enhanceAnim by remember { mutableStateOf<EnhanceAnim?>(null) }
    // Auto-capture waits for a steady live outline, so it needs live edge detection.
    var autoCapture by remember { mutableStateOf(autoCaptureDefault && liveOutline) }
    var steadyProgress by remember { mutableStateOf(0f) }

    // Set by the camera preview: focuses on the page.
    val focuser = remember { FocusHolder() }
    val motion = rememberMotionMonitor()

    /** Checks the new photo: page edges, enhanced preview for the animation, blur. */
    fun processPhoto(file: java.io.File, quadAtShutter: Quad?, shot: Long) {
        session.startedProcessing()
        scope.launch {
            val (q, sure) = withContext(detectDispatcher) {
                runCatching { ImageProcessor.detect(file) }.getOrElse { Quad.FULL to false }
            }
            // If the photo found nothing, use the outline that was on screen.
            val quad = if (q == Quad.FULL && quadAtShutter != null) quadAtShutter else q
            // The live outline was steady and agrees with the photo: that's a second opinion.
            val agrees = quadAtShutter != null && q.points.zip(quadAtShutter.points)
                .all { (p1, p2) -> abs(p1.x - p2.x) < 0.04f && abs(p1.y - p2.y) < 0.04f }
            val id = session.add(file, quad, sure || agrees)
            session.finishedProcessing()

            // The outline snaps onto the page in the animation, then the clean page is prepared.
            enhanceAnim?.let { a -> if (a.key == shot) enhanceAnim = a.copy(quad = quad) }
            val page = session.page(id) ?: return@launch
            launch {
                // The small cleaned page: for the animation and the thumbnail next to the shutter.
                val result = withContext(PreviewRenderer.dispatcher) {
                    runCatching {
                        val flat = PreviewRenderer.render(page, PreviewRenderer.SMALL_SIDE, ScanFilter.ORIGINAL)
                        flat.asImageBitmap() to PreviewRenderer.render(page, PreviewRenderer.SMALL_SIDE).asImageBitmap()
                    }.getOrNull()
                }
                val a = enhanceAnim
                if (a != null && a.key == shot) {
                    enhanceAnim = if (result != null) a.copy(flat = result.first, enhanced = result.second) else a.copy(failed = true)
                }
                // No new photo for a moment: quietly make this page's full preview, so the review
                // screen opens on it without waiting. Skipped while pages are being shot quickly.
                delay(1200)
                if (shotCounter == shot && !capturing) withContext(PreviewRenderer.dispatcher) {
                    runCatching { session.page(id)?.let { PreviewRenderer.render(it, PreviewRenderer.previewSide) } }
                }
            }
            // Blur check (in the background; never blocks the next photo).
            launch {
                val sharp = withContext(detectDispatcher) {
                    runCatching { ImageProcessor.sharpness(file, quad) }.getOrNull()
                }
                if (sharp != null && sharp < ImageProcessor.BLURRY && session.page(id) != null) blurryPage = id
            }
        }
    }

    fun takePhoto(quadAtShutter: Quad?) {
        val file = session.newPhotoFile()
        val shot = ++shotCounter
        // Freeze the picture right now: instant feedback, and the animation starts from it.
        val still = runCatching { focuser.previewView?.bitmap }.getOrNull()
        frozen = still
        scope.launch { frozenAlpha.snapTo(1f) }
        if (animate && still != null) enhanceAnim = EnhanceAnim(shot, still.asImageBitmap())
        stage = "Hold still…"
        imageCapture.takePicture(
            ImageCapture.OutputFileOptions.Builder(file).build(),
            ContextCompat.getMainExecutor(context),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    capturing = false
                    stage = null
                    if (!animate || enhanceAnim?.key != shot) {
                        // No animation: let the frozen picture fade back into the live camera.
                        scope.launch { frozenAlpha.animateTo(0f, tween(260)); if (shotCounter == shot) frozen = null }
                    }
                    processPhoto(file, quadAtShutter, shot)
                }

                override fun onError(exception: ImageCaptureException) {
                    capturing = false
                    stage = null
                    frozen = null
                    if (enhanceAnim?.key == shot) enhanceAnim = null
                    file.delete()
                    error = "Couldn't take the photo. Try again."
                }
            },
        )
    }

    fun capture() {
        if (capturing) return
        // Tapping the shutter while the animation plays skips it: the next page can be shot at once.
        if (enhanceAnim != null) { enhanceAnim = null; frozen = null }
        capturing = true
        error = null
        val quadAtShutter = liveQuad.value
        scope.launch {
            // 1. Focus on the page (sharp text). 2. Wait until the phone is still (no shake).
            stage = "Focusing…"
            focuser.focus(quadAtShutter)
            if (!motion.isStill()) {
                stage = "Hold still…"
                motion.awaitStill(maxWaitMs = 1200)
            }
            takePhoto(quadAtShutter)
        }
    }


    // Auto-capture: when the page outline stays still for about a second, take the photo.
    // After a shot, waits until the page moves (next page) before shooting again.
    LaunchedEffect(autoCapture, hasPermission) {
        if (!autoCapture || !hasPermission) return@LaunchedEffect
        var reference: Quad? = null
        var steadySince = 0L
        var needMove = false
        while (true) {
            delay(100)
            val q = liveQuad.value
            val now = SystemClock.elapsedRealtime()
            val busy = capturing || enhanceAnim != null || frozen != null
            if (q == null) { reference = null; steadyProgress = 0f; needMove = false; continue }
            val ref = reference
            val moved = ref == null || q.points.zip(ref.points).any { (a, b) -> abs(a.x - b.x) > 0.02f || abs(a.y - b.y) > 0.02f }
            if (moved) {
                if (ref != null && q.points.zip(ref.points).any { (a, b) -> abs(a.x - b.x) > 0.08f || abs(a.y - b.y) > 0.08f }) needMove = false
                reference = q; steadySince = now; steadyProgress = 0f
                continue
            }
            if (busy || needMove) { steadyProgress = 0f; continue }
            steadyProgress = ((now - steadySince) / 1200f).coerceIn(0f, 1f)
            if (steadyProgress >= 1f) {
                steadyProgress = 0f
                needMove = true
                capture()
            }
        }
    }

    val pickImages = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(30)) { uris ->
        if (uris.isEmpty()) return@rememberLauncherForActivityResult
        uris.forEach { _ -> session.startedProcessing() }
        scope.launch {
            for (uri in uris) {
                withContext(detectDispatcher) {
                    runCatching {
                        val file = session.newPhotoFile()
                        importGalleryImage(context, uri, file, session.outputMaxSide)
                        val (q, sure) = ImageProcessor.detect(file)
                        session.add(file, q, sure)
                    }
                }
                session.finishedProcessing()
            }
            onDone()
        }
    }

    Column(
        Modifier
            .fillMaxSize()
            .background(Color.Black)
    ) {
        // Top bar
        Row(
            Modifier.fillMaxWidth().statusBarsPadding().padding(horizontal = 4.dp, vertical = 4.dp),
            verticalAlignment = Alignment.CenterVertically,
        ) {
            IconButton(onClick = { leave() }) { Icon(Icons.Outlined.Close, "Close", tint = Color.White) }
            StatusText(
                fixed = if (!hasPermission) "Camera permission needed" else stage
                    ?: if (safeMode && liveOutline) "Safe mode: live outline off (tap to turn on)"
                    else if (!liveOutline) "Point at the page"
                    else null,
                liveQuad = liveQuad,
                modifier = Modifier.weight(1f).clickable(enabled = safeMode && liveOutline) {
                    com.myscanner.app.scan.NativeGuard.resetAll()
                    safeMode = false
                    error = "The live outline will be tried again the next time the camera opens."
                },
            )
            if (liveOutline) {
                TextButton(onClick = { autoCapture = !autoCapture; steadyProgress = 0f }) {
                    Text(
                        if (autoCapture) "AUTO" else "MANUAL",
                        color = if (autoCapture) Teal else Color.White,
                        style = MaterialTheme.typography.labelLarge,
                    )
                }
            }
            IconButton(onClick = {
                flashMode = when (flashMode) {
                    ImageCapture.FLASH_MODE_OFF -> ImageCapture.FLASH_MODE_AUTO
                    ImageCapture.FLASH_MODE_AUTO -> ImageCapture.FLASH_MODE_ON
                    else -> ImageCapture.FLASH_MODE_OFF
                }
            }) {
                val (icon, label) = when (flashMode) {
                    ImageCapture.FLASH_MODE_AUTO -> Icons.Outlined.FlashAuto to "Flash auto"
                    ImageCapture.FLASH_MODE_ON -> Icons.Outlined.FlashOn to "Flash on"
                    else -> Icons.Outlined.FlashOff to "Flash off"
                }
                Icon(icon, label, tint = Color.White)
            }
        }

        // Camera picture, exactly the shape of the photo (3:4), so what you see is what you get.
        Box(Modifier.weight(1f).fillMaxWidth(), contentAlignment = Alignment.Center) {
            Box(Modifier.fillMaxWidth().aspectRatio(3f / 4f)) {
                if (hasPermission) {
                    // The page finder rests while a photo is taken or its animation plays.
                    SideEffect { analyzer.paused = !liveOutline || safeMode || capturing || frozen != null || enhanceAnim != null }
                    CameraPreview(
                        imageCapture = imageCapture,
                        focuser = focuser,
                        // Live edge detection off: the page-finder stream isn't started at all.
                        analyzer = if (liveOutline) analyzer else null,
                        onError = { error = it },
                    )
                    if (liveOutline) PageOutline(liveQuad, hidden = capturing || frozen != null)
                } else {
                    Column(
                        Modifier.fillMaxSize().padding(32.dp),
                        verticalArrangement = Arrangement.Center,
                        horizontalAlignment = Alignment.CenterHorizontally,
                    ) {
                        Text(
                            "The app needs the camera to scan. Photos stay on this phone.",
                            color = Color.White,
                            textAlign = TextAlign.Center,
                        )
                        Spacer(Modifier.height(16.dp))
                        Button(onClick = { askPermission.launch(Manifest.permission.CAMERA) }) { Text("Allow camera") }
                    }
                }
                frozen?.let { bmp ->
                    if (enhanceAnim == null) {
                        Image(
                            bmp.asImageBitmap(), null,
                            contentScale = ContentScale.FillBounds,
                            modifier = Modifier.fillMaxSize(),
                            alpha = frozenAlpha.value,
                        )
                    }
                }
                enhanceAnim?.let { a -> EnhanceOverlay(a) { enhanceAnim = null; frozen = null } }
            }
        }

        blurryPage?.let { id ->
            LaunchedEffect(id) { delay(7000); if (blurryPage == id) blurryPage = null }
            Row(
                Modifier.fillMaxWidth().padding(horizontal = 12.dp, vertical = 4.dp)
                    .clip(RoundedCornerShape(12.dp)).background(Color(0xFF3A2E00)).padding(start = 14.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text("The last photo looks blurry", color = Color(0xFFFFE08A), modifier = Modifier.weight(1f),
                    style = MaterialTheme.typography.bodyMedium)
                TextButton(onClick = { session.remove(id); blurryPage = null }) { Text("Retake", color = Color.White) }
                TextButton(onClick = { blurryPage = null }) { Text("Keep", color = Color(0xFFBBBBBB)) }
            }
        }

        error?.let {
            Text(
                it,
                color = Color(0xFFFFB4AB),
                modifier = Modifier.fillMaxWidth().padding(8.dp).clickable { error = null },
                textAlign = TextAlign.Center,
            )
        }

        // Bottom controls
        Row(
            Modifier.fillMaxWidth().navigationBarsPadding().padding(horizontal = 24.dp, vertical = 20.dp),
            verticalAlignment = Alignment.CenterVertically,
            horizontalArrangement = Arrangement.SpaceBetween,
        ) {
            IconButton(
                onClick = { pickImages.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly)) },
                modifier = Modifier.size(56.dp),
            ) {
                Icon(Icons.Outlined.PhotoLibrary, "Import from gallery", tint = Color.White, modifier = Modifier.size(28.dp))
            }

            // Shutter, with a ring that fills while auto-capture waits for the page to be steady
            Box(contentAlignment = Alignment.Center) {
                if (autoCapture && steadyProgress > 0f) {
                    CircularProgressIndicator(
                        progress = { steadyProgress },
                        modifier = Modifier.size(90.dp),
                        color = Teal,
                        strokeWidth = 4.dp,
                    )
                }
                Box(
                    Modifier
                        .size(78.dp)
                        .border(4.dp, Color.White, CircleShape)
                        .padding(8.dp)
                        .clip(CircleShape)
                        .background(if (capturing || !hasPermission) Color.Gray else Color.White)
                        .clickable(enabled = hasPermission && !capturing, onClickLabel = "Take photo") { capture() },
                )
            }

            LastPageButton(
                session = session,
                count = pages.size,
                busy = processing > 0,
                onClick = { if (pages.isNotEmpty() && processing == 0) onDone() },
            )
        }
    }

    if (showDiscard) {
        ConfirmDialog(
            title = "Discard ${pages.size} page(s)?",
            message = "The pages you scanned haven't been saved yet.",
            confirmLabel = "Discard",
            onConfirm = { showDiscard = false; onClose() },
            onDismiss = { showDiscard = false },
        )
    }
}

@Composable
private fun CameraPreview(
    imageCapture: ImageCapture,
    focuser: FocusHolder,
    /** Null: no live page finder (only the preview and the photo streams). */
    analyzer: EdgeAnalyzer?,
    onError: (String) -> Unit,
) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    var camera by remember { mutableStateOf<Camera?>(null) }
    val previewView = remember {
        PreviewView(context).apply {
            scaleType = PreviewView.ScaleType.FIT_CENTER
            // TextureView: no black flashes or flicker when things are drawn on top of the camera,
            // and the picture can be frozen instantly at the moment of the shot.
            implementationMode = PreviewView.ImplementationMode.COMPATIBLE
        }
    }

    DisposableEffect(lifecycleOwner, imageCapture, analyzer) {
        val executor = Executors.newSingleThreadExecutor()
        val future = ProcessCameraProvider.getInstance(context)
        var provider: ProcessCameraProvider? = null
        future.addListener({
            try {
                val p = future.get()
                provider = p
                val aspect = ResolutionSelector.Builder()
                    .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                    .build()
                val preview = Preview.Builder().setResolutionSelector(aspect).build()
                preview.setSurfaceProvider(previewView.surfaceProvider)
                val analysis = if (analyzer == null) null else ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .setResolutionSelector(
                        ResolutionSelector.Builder()
                            .setAspectRatioStrategy(AspectRatioStrategy.RATIO_4_3_FALLBACK_AUTO_STRATEGY)
                            .setResolutionStrategy(
                                ResolutionStrategy(Size(640, 480), ResolutionStrategy.FALLBACK_RULE_CLOSEST_LOWER_THEN_HIGHER)
                            )
                            .build()
                    )
                    .build()
                analysis?.setAnalyzer(executor, analyzer!!)
                p.unbindAll()
                camera = if (analysis == null) {
                    p.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
                } else try {
                    p.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture, analysis)
                } catch (e: Exception) {
                    // Some older phones can't run three camera streams: scan without the live outline.
                    p.unbindAll()
                    p.bindToLifecycle(lifecycleOwner, CameraSelector.DEFAULT_BACK_CAMERA, preview, imageCapture)
                }
                focuser.camera = camera
                focuser.previewView = previewView
            } catch (e: Exception) {
                onError("Couldn't start the camera: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            focuser.camera = null
            provider?.unbindAll()
            executor.shutdown()
        }
    }

    AndroidView(
        factory = { previewView },
        modifier = Modifier
            .fillMaxSize()
            .pointerInput(Unit) {
                // Tap to focus on that spot: sharper text.
                detectTapGestures { offset ->
                    val point = previewView.meteringPointFactory.createPoint(offset.x, offset.y)
                    camera?.cameraControl?.startFocusAndMetering(FocusMeteringAction.Builder(point).build())
                }
            },
    )
}

/** The live page outline drawn over the camera picture. Glides smoothly between positions. */
@Composable
private fun PageOutline(liveQuad: StateFlow<Quad?>, hidden: Boolean) {
    val live by liveQuad.collectAsStateWithLifecycle()
    val quad = if (hidden) null else live
    var last by remember { mutableStateOf(Quad.FULL) }
    LaunchedEffect(quad) { if (quad != null) last = quad }
    val target = quad ?: last
    val spec = tween<Float>(durationMillis = 170, easing = LinearOutSlowInEasing)
    val alpha by animateFloatAsState(if (quad != null) 1f else 0f, tween(220), label = "outlineAlpha")
    val xs = target.points.mapIndexed { i, p -> animateFloatAsState(p.x, spec, label = "x$i") }
    val ys = target.points.mapIndexed { i, p -> animateFloatAsState(p.y, spec, label = "y$i") }
    androidx.compose.foundation.Canvas(Modifier.fillMaxSize()) {
        if (alpha <= 0.01f) return@Canvas
        val pts = (0 until 4).map { Offset(xs[it].value * size.width, ys[it].value * size.height) }
        val path = Path().apply {
            moveTo(pts[0].x, pts[0].y)
            for (i in 1 until 4) lineTo(pts[i].x, pts[i].y)
            close()
        }
        drawPath(path, Teal.copy(alpha = 0.18f * alpha))
        drawPath(path, Teal.copy(alpha = alpha), style = Stroke(width = 3.dp.toPx()))
        pts.forEach { drawCircle(Teal.copy(alpha = alpha), radius = 6.dp.toPx(), center = it) }
    }
}

/** The line at the top: what the camera is doing, or whether a page is in view. */
@Composable
private fun StatusText(fixed: String?, liveQuad: StateFlow<Quad?>, modifier: Modifier = Modifier) {
    val found = if (fixed == null) liveQuad.collectAsStateWithLifecycle().value != null else false
    Text(
        fixed ?: if (found) "Page found – hold steady" else "Point at the page",
        color = Color.White,
        style = MaterialTheme.typography.titleSmall,
        textAlign = TextAlign.Center,
        modifier = modifier,
    )
}

/** Thumbnail of the last page with the page count. Tap to review and save. */
@Composable
private fun LastPageButton(session: ScanSession, count: Int, busy: Boolean, onClick: () -> Unit) {
    val pages by session.pages.collectAsStateWithLifecycle()
    val last = pages.lastOrNull()
    var thumb by remember { mutableStateOf<android.graphics.Bitmap?>(null) }
    LaunchedEffect(last) {
        thumb = if (last == null) null else withContext(PreviewRenderer.dispatcher) {
            runCatching { PreviewRenderer.render(last, 160) }.getOrNull()
        }
    }
    BadgedBox(
        badge = { if (count > 0) Badge { Text("$count") } },
        modifier = Modifier.size(56.dp),
    ) {
        Box(
            Modifier
                .size(56.dp)
                .clip(RoundedCornerShape(10.dp))
                .background(Color.DarkGray)
                .clickable(enabled = count > 0 && !busy, onClickLabel = "Review and save") { onClick() },
            contentAlignment = Alignment.Center,
        ) {
            val bmp = thumb
            if (bmp != null) {
                Image(bmp.asImageBitmap(), "Review $count pages", contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
            }
            if (busy) CircularProgressIndicator(Modifier.size(24.dp), color = Color.White, strokeWidth = 2.dp)
            else if (count > 0) Text("Done", color = Color.White, style = MaterialTheme.typography.labelMedium,
                modifier = Modifier.align(Alignment.BottomCenter).fillMaxWidth().background(Color.Black.copy(alpha = 0.55f)).padding(2.dp),
                textAlign = TextAlign.Center)
        }
    }
}

/**
 * Focuses the camera on the page (its centre, or the middle of the picture). Focus only:
 * the brightness is left alone, so the picture doesn't flicker before the shot. Never waits
 * more than 1.2 s.
 */
class FocusHolder {
    var camera: Camera? = null
    var previewView: PreviewView? = null

    suspend fun focus(quad: Quad?) {
        val cam = camera ?: return
        val view = previewView ?: return
        if (view.width == 0) return
        val cx = quad?.points?.map { it.x }?.average()?.toFloat() ?: 0.5f
        val cy = quad?.points?.map { it.y }?.average()?.toFloat() ?: 0.5f
        withTimeoutOrNull(1200) {
            suspendCancellableCoroutine<Unit> { cont ->
                try {
                    val point = view.meteringPointFactory.createPoint(cx * view.width, cy * view.height)
                    val request = FocusMeteringAction.Builder(point, FocusMeteringAction.FLAG_AF)
                        .setAutoCancelDuration(4, TimeUnit.SECONDS)
                        .build()
                    val future = cam.cameraControl.startFocusAndMetering(request)
                    future.addListener({ if (cont.isActive) cont.resume(Unit) }, ContextCompat.getMainExecutor(view.context))
                } catch (e: Exception) {
                    if (cont.isActive) cont.resume(Unit)
                }
            }
        }
    }
}

/**
 * Watches the phone's gyroscope, so the photo is taken only when the hands are still: the main
 * cause of blurry scans. Phones without a gyroscope use the accelerometer; without either it
 * never waits.
 */
class MotionMonitor(context: android.content.Context) : SensorEventListener {
    private val manager = context.getSystemService(android.content.Context.SENSOR_SERVICE) as? SensorManager
    private val sensor: Sensor? = manager?.getDefaultSensor(Sensor.TYPE_GYROSCOPE)
        ?: manager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
    private val gyro = sensor?.type == Sensor.TYPE_GYROSCOPE
    @Volatile private var lastShake = 0L
    private var lastAcc: FloatArray? = null

    fun start() { sensor?.let { manager?.registerListener(this, it, SensorManager.SENSOR_DELAY_GAME) } }
    fun stop() { manager?.unregisterListener(this) }

    override fun onSensorChanged(event: SensorEvent) {
        val v = event.values
        val moving = if (gyro) {
            sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]) > 0.10f // radians per second
        } else {
            val prev = lastAcc
            lastAcc = v.clone()
            prev != null && sqrt((v[0] - prev[0]).let { it * it } + (v[1] - prev[1]).let { it * it } + (v[2] - prev[2]).let { it * it }) > 0.25f
        }
        if (moving) lastShake = SystemClock.elapsedRealtime()
    }

    override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) = Unit

    /** True when the phone hasn't moved for the last 0.25 s. */
    fun isStill(): Boolean = sensor == null || SystemClock.elapsedRealtime() - lastShake > 250

    suspend fun awaitStill(maxWaitMs: Long) {
        val start = SystemClock.elapsedRealtime()
        while (!isStill() && SystemClock.elapsedRealtime() - start < maxWaitMs) delay(30)
    }
}

@Composable
private fun rememberMotionMonitor(): MotionMonitor {
    val context = LocalContext.current
    val monitor = remember { MotionMonitor(context.applicationContext) }
    DisposableEffect(monitor) {
        monitor.start()
        onDispose { monitor.stop() }
    }
    return monitor
}
