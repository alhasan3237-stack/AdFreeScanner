package com.myscanner.app.ui.scan

import android.graphics.Bitmap
import androidx.activity.compose.BackHandler
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
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
import androidx.compose.foundation.horizontalScroll
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.AddAPhoto
import androidx.compose.material.icons.outlined.Crop
import androidx.compose.material.icons.outlined.CreditCard
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Draw
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.automirrored.outlined.ArrowForward
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material.icons.outlined.RotateRight
import androidx.compose.material.icons.outlined.WarningAmber
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.Scaffold
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.myscanner.app.ScannerApp
import com.myscanner.app.data.AppSettings
import com.myscanner.app.data.ScanFilter
import com.myscanner.app.scan.PreviewRenderer
import com.myscanner.app.scan.ScanPage
import com.myscanner.app.scan.ScanSession
import com.myscanner.app.ui.components.ConfirmDialog
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.first
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext

/** Saving runs here so it isn't interrupted if the screen is redrawn. */
class SaveViewModel(private val app: ScannerApp) : ViewModel() {
    /** (pages done, total) while saving; null otherwise. */
    private val _progress = MutableStateFlow<Pair<Int, Int>?>(null)
    val progress: StateFlow<Pair<Int, Int>?> = _progress.asStateFlow()

    /** True once saved: the screen is about to close. */
    private val _saved = MutableStateFlow(false)
    val saved: StateFlow<Boolean> = _saved.asStateFlow()

    private val _error = MutableStateFlow<String?>(null)
    val error: StateFlow<String?> = _error.asStateFlow()

    fun save(session: ScanSession, onSaved: (docId: Long, wasNew: Boolean) -> Unit) {
        if (_progress.value != null) return
        _progress.value = 0 to session.pages.value.size
        viewModelScope.launch {
            try {
                val settings: AppSettings = app.settings.settings.first()
                val wasNew = session.targetDocumentId == null && !session.isEditing
                val id = app.repository.saveScanSession(session, settings.smartNames, settings.keepOriginals) { done, total ->
                    _progress.value = done to total
                }
                _saved.value = true
                app.endSession()
                onSaved(id, wasNew)
            } catch (e: Throwable) {
                _error.value = "Couldn't save: ${e.message ?: "out of memory"}"
            } finally {
                _progress.value = null
            }
        }
    }

    fun errorShown() { _error.value = null }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun ReviewScreen(
    app: ScannerApp,
    session: ScanSession,
    onAddMore: () -> Unit,
    onCrop: (pageId: Long) -> Unit,
    /** Opens the drawing screen (pen, highlighter, text, signature) for a page. */
    onMarkup: (pageId: Long) -> Unit,
    onSaved: (docId: Long, wasNew: Boolean) -> Unit,
    onDiscard: () -> Unit,
) {
    val vm: SaveViewModel = viewModel { SaveViewModel(app) }
    val pages by session.pages.collectAsStateWithLifecycle()
    val progress by vm.progress.collectAsStateWithLifecycle()
    val error by vm.error.collectAsStateWithLifecycle()
    val saved by vm.saved.collectAsStateWithLifecycle()
    // Opens on the newest photo (the one just taken). The order of the pages doesn't change:
    // swipe back for the older ones; the PDF still starts with the first photo.
    val pagerState = rememberPagerState(
        initialPage = (session.openAtPageIndex ?: (pages.size - 1)).coerceIn(0, (pages.size - 1).coerceAtLeast(0)),
        pageCount = { pages.size },
    )
    var showMore by remember { mutableStateOf(false) }
    var knownCount by rememberSaveable { mutableStateOf(pages.size) }
    LaunchedEffect(pages.size) {
        // A photo that was still being processed arrived: show it.
        if (pages.size > knownCount && pages.isNotEmpty()) pagerState.scrollToPage(pages.size - 1)
        knownCount = pages.size
    }
    LaunchedEffect(pagerState.currentPage, pages) {
        kotlinx.coroutines.delay(350) // let the page on screen go first
        for (i in listOf(pagerState.currentPage + 1, pagerState.currentPage - 1)) {
            val p = pages.getOrNull(i) ?: continue
            withContext(PreviewRenderer.dispatcher) { runCatching { PreviewRenderer.render(p, PreviewRenderer.previewSide) } }
        }
    }
    var showDelete by remember { mutableStateOf(false) }
    var zoomed by remember { mutableStateOf(false) }
    var idHint by remember { mutableStateOf(false) }
    var idBusy by remember { mutableStateOf(false) }
    val idScope = rememberCoroutineScope()
    LaunchedEffect(pagerState.currentPage) { zoomed = false }
    var showDiscard by remember { mutableStateOf(false) }

    // Scanning: back = back to the camera. Editing a saved document: back = leave (asks first).
    BackHandler { if (session.isEditing) showDiscard = true else onAddMore() }

    if (pages.isEmpty()) {
        // Every page was deleted: back to the camera. (After saving, the pages are gone too: stay put.)
        // Editing and every page was removed: leave; the saved document stays as it was.
        if (!saved && progress == null) LaunchedEffect(Unit) { if (session.isEditing) onDiscard() else onAddMore() }
        return
    }
    val current: ScanPage? = pages.getOrNull(pagerState.currentPage.coerceIn(0, pages.size - 1))

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Page ${pagerState.currentPage + 1} of ${pages.size}") },
                navigationIcon = {
                    IconButton(onClick = { if (session.isEditing) showDiscard = true else onAddMore() }) {
                        Icon(Icons.AutoMirrored.Outlined.ArrowBack, if (session.isEditing) "Back" else "Back to camera")
                    }
                },
                actions = {
                    if (!session.isEditing) TextButton(onClick = { showDiscard = true }) { Text("Discard") }
                    Button(onClick = { vm.save(session, onSaved) }) { Text(if (session.isEditing) "Save changes" else "Save") }
                    Box {
                        IconButton(onClick = { showMore = true }) { Icon(Icons.Outlined.MoreVert, "More tools") }
                        DropdownMenu(expanded = showMore, onDismissRequest = { showMore = false }) {
                            val i = pagerState.currentPage
                            DropdownMenuItem(
                                text = { Text("Move page earlier") },
                                leadingIcon = { Icon(Icons.AutoMirrored.Outlined.ArrowBack, null) },
                                enabled = i > 0,
                                onClick = {
                                    showMore = false
                                    current?.let { session.move(it.id, -1); idScope.launch { pagerState.scrollToPage(i - 1) } }
                                },
                            )
                            DropdownMenuItem(
                                text = { Text("Move page later") },
                                leadingIcon = { Icon(Icons.AutoMirrored.Outlined.ArrowForward, null) },
                                enabled = i < pages.size - 1,
                                onClick = {
                                    showMore = false
                                    current?.let { session.move(it.id, 1); idScope.launch { pagerState.scrollToPage(i + 1) } }
                                },
                            )
                            DropdownMenuItem(
                                text = { Text("ID card: 2 sides on one page") },
                                leadingIcon = { Icon(Icons.Outlined.CreditCard, null) },
                                onClick = {
                                    showMore = false
                                    val front = pages.getOrNull(i)
                                    val back = pages.getOrNull(i + 1)
                                    if (front == null || back == null) {
                                        idHint = true
                                    } else {
                                        idBusy = true
                                        idScope.launch {
                                            val ok = withContext(PreviewRenderer.dispatcher) {
                                                runCatching {
                                                    com.myscanner.app.scan.HeavyWork.lock.withLock {
                                                        val f = com.myscanner.app.scan.ImageProcessor.render(front.file, front.quad, front.filter, front.rotation, 1400, front.marks)
                                                        val b = com.myscanner.app.scan.ImageProcessor.render(back.file, back.quad, back.filter, back.rotation, 1400, back.marks)
                                                        val dest = session.newPhotoFile()
                                                        com.myscanner.app.scan.ImageProcessor.composeIdCard(f, b, dest)
                                                        f.release(); b.release()
                                                        dest
                                                    }
                                                }.getOrNull()
                                            }
                                            if (ok != null) session.replaceWithCombined(front.id, back.id, ok)
                                            idBusy = false
                                        }
                                    }
                                },
                            )
                        }
                    }
                },
            )
        },
        bottomBar = {
            Surface(tonalElevation = 3.dp) {
                Column(Modifier.fillMaxWidth().navigationBarsPadding().padding(top = 10.dp, bottom = 6.dp)) {
                    if (current != null) {
                        FilterStrip(
                            page = current,
                            onPick = { f -> session.update(current.id) { it.copy(filter = f) } },
                        )
                        if (pages.size > 1) {
                            TextButton(
                                onClick = { session.setFilterForAll(current.filter) },
                                modifier = Modifier.align(Alignment.CenterHorizontally),
                            ) { Text("Use “${current.filter.label}” on all ${pages.size} pages") }
                        }
                    }
                    Row(
                        Modifier.fillMaxWidth().padding(horizontal = 8.dp),
                        horizontalArrangement = Arrangement.SpaceEvenly,
                    ) {
                        Tool("Crop", { Icon(Icons.Outlined.Crop, null) }) { current?.let { onCrop(it.id) } }
                        Tool("Rotate", { Icon(Icons.Outlined.RotateRight, null) }) { current?.let { session.rotate(it.id) } }
                        Tool("Draw / sign", { Icon(Icons.Outlined.Draw, null) }) { current?.let { onMarkup(it.id) } }
                        Tool("Add page", { Icon(Icons.Outlined.AddAPhoto, null) }) { onAddMore() }
                        Tool("Delete", { Icon(Icons.Outlined.Delete, null, tint = MaterialTheme.colorScheme.error) }) {
                            showDelete = true
                        }
                    }
                }
            }
        },
    ) { padding ->
        HorizontalPager(
            state = pagerState,
            userScrollEnabled = !zoomed, // while zoomed in, dragging moves around the page instead
            key = { i -> pages.getOrNull(i)?.id ?: i },
            modifier = Modifier.padding(padding).fillMaxSize().background(MaterialTheme.colorScheme.surfaceContainer),
        ) { index ->
            pages.getOrNull(index)?.let { page ->
                PagePreview(
                    page = page,
                    outputMaxSide = session.outputMaxSide,
                    isCurrent = index == pagerState.currentPage,
                    onZoomed = { if (index == pagerState.currentPage) zoomed = it },
                )
            }
        }
    }

    if (showDelete && current != null) {
        ConfirmDialog(
            title = "Delete this page?",
            message = "It will be removed from this scan.",
            confirmLabel = "Delete",
            onConfirm = { showDelete = false; session.remove(current.id) },
            onDismiss = { showDelete = false },
        )
    }
    if (showDiscard) {
        ConfirmDialog(
            title = if (session.isEditing) "Leave without saving?" else "Discard this scan?",
            message = if (session.isEditing) "Your changes to these pages won't be kept. The saved document stays as it was."
                      else "All ${pages.size} page(s) will be thrown away.",
            confirmLabel = if (session.isEditing) "Leave" else "Discard",
            onConfirm = { showDiscard = false; onDiscard() },
            onDismiss = { showDiscard = false },
        )
    }
    progress?.let { (done, total) ->
        AlertDialog(
            onDismissRequest = {},
            confirmButton = {},
            title = { Text("Saving…") },
            text = {
                Column {
                    Text(if (done < total) "Cleaning page ${done + 1} of $total" else "Finishing…")
                    Spacer(Modifier.height(12.dp))
                    LinearProgressIndicator(progress = { if (total == 0) 0f else done.toFloat() / total }, modifier = Modifier.fillMaxWidth())
                }
            },
        )
    }
    if (idHint) {
        AlertDialog(
            onDismissRequest = { idHint = false },
            confirmButton = { TextButton(onClick = { idHint = false }) { Text("OK") } },
            title = { Text("ID card") },
            text = { Text("Scan the front of the card, then the back as the next page. Open the front and tap ID card: both sides go on one A4 page at real card size.") },
        )
    }
    if (idBusy) com.myscanner.app.ui.components.BusyDialog("Making the ID card page…")
    error?.let {
        AlertDialog(
            onDismissRequest = vm::errorShown,
            confirmButton = { TextButton(onClick = vm::errorShown) { Text("OK") } },
            text = { Text(it) },
        )
    }
}

/** One page, straightened and filtered. The old picture stays on screen until the new one is ready. */
@Composable
private fun PagePreview(page: ScanPage, outputMaxSide: Int, isCurrent: Boolean, onZoomed: (Boolean) -> Unit) {
    var bitmap by remember(page.id) { mutableStateOf<Bitmap?>(null) }
    var sharp by remember(page.id) { mutableStateOf<Bitmap?>(null) } // full detail, made when zoomed in
    var zoom by remember(page.id) { mutableStateOf(1f) }
    var loading by remember { mutableStateOf(true) }
    LaunchedEffect(page.quad, page.filter, page.rotation, page.marks) {
        loading = true
        sharp = null
        bitmap = withContext(PreviewRenderer.dispatcher) {
            runCatching { PreviewRenderer.render(page, PreviewRenderer.previewSide) }.getOrNull()
        } ?: bitmap
        loading = false
    }
    // Zoomed in: make a sharp, high-detail version (not cached, freed when the page changes).
    val wantSharp = zoom > 1.3f && isCurrent
    LaunchedEffect(wantSharp, page.quad, page.filter, page.rotation, page.marks) {
        if (wantSharp && sharp == null) {
            sharp = withContext(PreviewRenderer.dispatcher) {
                runCatching {
                    com.myscanner.app.scan.HeavyWork.lock.withLock {
                        val mat = com.myscanner.app.scan.ImageProcessor.render(
                            page.file, page.quad, page.filter, page.rotation, minOf(outputMaxSide, 2400), page.marks
                        )
                        com.myscanner.app.scan.ImageProcessor.toBitmapSmall(mat).also { mat.release() }
                    }
                }.getOrNull()
            }
        }
    }
    Box(Modifier.fillMaxSize().padding(16.dp), contentAlignment = Alignment.Center) {
        com.myscanner.app.ui.components.ZoomableBox(
            modifier = Modifier.fillMaxSize(),
            resetKey = page.id to page.rotation,
            onZoom = { z -> zoom = z; onZoomed(z > 1.01f) },
        ) {
            (sharp ?: bitmap)?.let {
                Image(it.asImageBitmap(), "Scanned page", contentScale = ContentScale.Fit, modifier = Modifier.fillMaxSize())
            }
        }
        if (loading && bitmap == null) CircularProgressIndicator()
        if (!page.detected) {
            Row(
                Modifier
                    .align(Alignment.TopCenter)
                    .clip(RoundedCornerShape(12.dp))
                    .background(MaterialTheme.colorScheme.surface.copy(alpha = 0.9f))
                    .padding(horizontal = 12.dp, vertical = 6.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Outlined.WarningAmber, null, Modifier.size(18.dp))
                Spacer(Modifier.width(6.dp))
                Text("Please check the corners – tap Crop", style = MaterialTheme.typography.bodySmall)
            }
        }
    }
}

/** Small previews of each filter for the current page. */
@Composable
private fun FilterStrip(page: ScanPage, onPick: (ScanFilter) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .horizontalScroll(rememberScrollState())
            .padding(horizontal = 12.dp),
        horizontalArrangement = Arrangement.spacedBy(6.dp),
    ) {
        ScanFilter.entries.forEach { filter ->
            var thumb by remember(page.id, page.quad, page.rotation, filter) { mutableStateOf<Bitmap?>(null) }
            var failed by remember(page.id, page.quad, page.rotation, filter) { mutableStateOf(false) }
            LaunchedEffect(page.id, page.quad, page.rotation, filter) {
                thumb = withContext(PreviewRenderer.dispatcher) {
                    runCatching { PreviewRenderer.render(page, 180, filter, withMarks = false) }.getOrNull()
                }
                failed = thumb == null
            }
            val selected = filter == page.filter
            Column(
                horizontalAlignment = Alignment.CenterHorizontally,
                modifier = Modifier
                    .clip(RoundedCornerShape(10.dp))
                    .clickable(onClickLabel = "Use ${filter.label}") { onPick(filter) }
                    .padding(4.dp),
            ) {
                Box(
                    Modifier
                        .size(width = 60.dp, height = 76.dp)
                        .clip(RoundedCornerShape(8.dp))
                        .border(
                            BorderStroke(
                                if (selected) 3.dp else 1.dp,
                                if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.outlineVariant,
                            ),
                            RoundedCornerShape(8.dp),
                        )
                        .background(MaterialTheme.colorScheme.surfaceContainerLowest),
                    contentAlignment = Alignment.Center,
                ) {
                    val t = thumb
                    if (t != null) Image(t.asImageBitmap(), null, contentScale = ContentScale.Crop, modifier = Modifier.fillMaxSize())
                    else if (!failed) CircularProgressIndicator(Modifier.size(18.dp), strokeWidth = 2.dp)
                }
                Text(
                    filter.label,
                    style = MaterialTheme.typography.labelMedium,
                    fontWeight = if (selected) FontWeight.Bold else FontWeight.Normal,
                    color = if (selected) MaterialTheme.colorScheme.primary else MaterialTheme.colorScheme.onSurface,
                    modifier = Modifier.padding(top = 4.dp),
                )
            }
        }
    }
}

@Composable
private fun Tool(label: String, icon: @Composable () -> Unit, onClick: () -> Unit) {
    TextButton(onClick = onClick) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            icon()
            Text(label, style = MaterialTheme.typography.labelMedium)
        }
    }
}
