package com.myscanner.app.ui.page

import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.pager.HorizontalPager
import androidx.compose.foundation.pager.rememberPagerState
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.RotateLeft
import androidx.compose.material.icons.outlined.RotateRight
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
import androidx.compose.runtime.mutableFloatStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.asImageBitmap
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.unit.dp
import coil.compose.AsyncImage
import com.myscanner.app.ScannerApp
import com.myscanner.app.data.PageEntity
import com.myscanner.app.data.pages
import com.myscanner.app.ui.components.BusyDialog
import com.myscanner.app.ui.components.ConfirmDialog
import com.myscanner.app.ui.components.ZoomableBox
import kotlinx.coroutines.flow.collectLatest
import kotlinx.coroutines.launch
import java.io.File

/**
 * A document's pages full screen, starting at the page that was tapped. Swipe for the other
 * pages, pinch to zoom, rotate or delete the page on screen.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PageScreen(
    app: ScannerApp,
    pageId: Long,
    onBack: () -> Unit,
    onDocumentDeleted: () -> Unit,
    /** Opens the document for editing (crop, filters, drawing...) on the page being shown. */
    onEdit: (documentId: Long, pageId: Long) -> Unit = { _, _ -> },
) {
    val repository = app.repository
    val scope = rememberCoroutineScope()
    var pages by remember { mutableStateOf<List<PageEntity>?>(null) }
    var busy by remember { mutableStateOf<String?>(null) }
    var showDelete by remember { mutableStateOf(false) }

    // Follow the page's document so rotations and deletions show up straight away.
    LaunchedEffect(pageId) {
        val first = repository.getPage(pageId) ?: run { onBack(); return@LaunchedEffect }
        repository.observeDocument(first.documentId).collectLatest { doc -> pages = doc?.pages.orEmpty() }
    }

    val list = pages
    if (list == null) {
        Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.surfaceContainer))
        return
    }
    if (list.isEmpty()) {
        LaunchedEffect(Unit) { onBack() }
        return
    }
    val startIndex = remember { list.indexOfFirst { it.id == pageId }.coerceAtLeast(0) }
    val pagerState = rememberPagerState(initialPage = startIndex) { list.size }
    val index = pagerState.currentPage.coerceIn(0, list.size - 1)
    val current = list[index]
    var zoomed by remember { mutableStateOf(false) }
    LaunchedEffect(pagerState.currentPage) { zoomed = false }

    fun rotate(degrees: Int) = scope.launch {
        if (busy != null) return@launch
        busy = "Rotating…"
        try { repository.rotatePage(current.id, degrees) } catch (e: Exception) { /* the page stays as it was */ } finally { busy = null }
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Page ${index + 1} of ${list.size}") },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Outlined.ArrowBack, "Back") }
                },
            )
        },
        bottomBar = {
            Surface(tonalElevation = 3.dp) {
                Row(
                    Modifier.fillMaxWidth().navigationBarsPadding().padding(8.dp),
                    horizontalArrangement = Arrangement.SpaceEvenly,
                ) {
                    ToolButton("Edit", { Icon(Icons.Outlined.Edit, null) }) { onEdit(current.documentId, current.id) }
                    ToolButton("Rotate left", { Icon(Icons.Outlined.RotateLeft, null) }) { rotate(-90) }
                    ToolButton("Rotate right", { Icon(Icons.Outlined.RotateRight, null) }) { rotate(90) }
                    ToolButton(
                        "Delete page",
                        { Icon(Icons.Outlined.Delete, null, tint = MaterialTheme.colorScheme.error) },
                    ) { showDelete = true }
                }
            }
        },
    ) { padding ->
        HorizontalPager(
            state = pagerState,
            userScrollEnabled = !zoomed, // while zoomed in, dragging moves around the page instead
            key = { i -> list.getOrNull(i)?.id ?: i },
            modifier = Modifier
                .padding(padding)
                .fillMaxSize()
                .background(MaterialTheme.colorScheme.surfaceContainer),
        ) { i ->
            list.getOrNull(i)?.let { page ->
                PageView(
                    page = page,
                    number = i + 1,
                    isCurrent = i == pagerState.currentPage,
                    onZoomed = { if (i == pagerState.currentPage) zoomed = it },
                )
            }
        }
    }

    if (showDelete) {
        ConfirmDialog(
            title = "Delete this page?",
            message = "The page will be removed from the document.",
            confirmLabel = "Delete",
            onConfirm = {
                showDelete = false
                val id = current.id
                scope.launch {
                    val documentGone = repository.deletePage(id)
                    if (documentGone) onDocumentDeleted() // otherwise the next page slides in
                }
            },
            onDismiss = { showDelete = false },
        )
    }
    busy?.let { BusyDialog(it) }
}

/** One page: pinch or double-tap to zoom; sharp full-detail image while zoomed. */
@Composable
private fun PageView(page: PageEntity, number: Int, isCurrent: Boolean, onZoomed: (Boolean) -> Unit) {
    var zoom by remember(page.imagePath) { mutableFloatStateOf(1f) }
    var sharp by remember(page.imagePath) { mutableStateOf<android.graphics.Bitmap?>(null) }
    val wantSharp = zoom > 1.3f && isCurrent
    LaunchedEffect(wantSharp, page.imagePath) {
        if (wantSharp && sharp == null) {
            sharp = kotlinx.coroutines.withContext(kotlinx.coroutines.Dispatchers.IO) { decodeSharp(page.imagePath) }
        }
    }
    ZoomableBox(
        modifier = Modifier.fillMaxSize(),
        resetKey = page.imagePath to isCurrent,
        onZoom = { z -> zoom = z; onZoomed(z > 1.01f) },
    ) {
        val hi = sharp
        if (hi != null) {
            Image(hi.asImageBitmap(), "Page $number", contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize().padding(16.dp))
        } else {
            AsyncImage(
                model = File(page.imagePath),
                contentDescription = "Page $number",
                contentScale = ContentScale.Fit,
                modifier = Modifier.fillMaxSize().padding(16.dp),
            )
        }
    }
}

/** Decodes a page at up to 2400 px (sharp when zoomed, safe for the GPU and memory). */
private fun decodeSharp(path: String): android.graphics.Bitmap? = runCatching {
    val o = android.graphics.BitmapFactory.Options().apply { inJustDecodeBounds = true }
    android.graphics.BitmapFactory.decodeFile(path, o)
    var sample = 1
    while (maxOf(o.outWidth, o.outHeight) / sample > 2400) sample *= 2
    android.graphics.BitmapFactory.decodeFile(path, android.graphics.BitmapFactory.Options().apply {
        inSampleSize = sample
        inPreferredConfig = android.graphics.Bitmap.Config.RGB_565 // half the memory, fine for documents
    })
}.getOrNull()

@Composable
private fun ToolButton(label: String, icon: @Composable () -> Unit, onClick: () -> Unit) {
    TextButton(onClick = onClick) {
        Column(horizontalAlignment = Alignment.CenterHorizontally) {
            icon()
            Text(label, style = MaterialTheme.typography.labelMedium)
        }
    }
}
