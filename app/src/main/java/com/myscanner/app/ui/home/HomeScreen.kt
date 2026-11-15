package com.myscanner.app.ui.home

import android.content.Context
import android.net.Uri
import android.provider.OpenableColumns
import androidx.activity.compose.BackHandler
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.combinedClickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.CheckCircle
import androidx.compose.material.icons.outlined.Close
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.DocumentScanner
import androidx.compose.material.icons.outlined.MergeType
import androidx.compose.material.icons.outlined.PictureAsPdf
import androidx.compose.material.icons.outlined.Search
import androidx.compose.material.icons.outlined.Settings
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.ExtendedFloatingActionButton
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TextField
import androidx.compose.material3.TextFieldDefaults
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.myscanner.app.ScannerApp
import com.myscanner.app.data.DocumentWithPages
import com.myscanner.app.data.pages
import com.myscanner.app.data.FileStore
import com.myscanner.app.data.HomeSort
import androidx.activity.result.PickVisualMediaRequest
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items as gridItems
import androidx.compose.material.icons.automirrored.outlined.Sort
import androidx.compose.material.icons.outlined.Check
import androidx.compose.material.icons.outlined.FileOpen
import androidx.compose.material.icons.outlined.GridView
import androidx.compose.material.icons.outlined.PhotoLibrary
import androidx.compose.material.icons.outlined.ViewList
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.HorizontalDivider
import com.myscanner.app.ui.components.BusyDialog
import com.myscanner.app.ui.components.ConfirmDialog
import com.myscanner.app.ui.components.TextInputDialog
import com.myscanner.app.ui.components.shareText
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun HomeScreen(
    app: ScannerApp,
    startScan: () -> Unit,
    /** Photos picked from the gallery become a new scan. */
    importPhotos: (List<Uri>) -> Unit,
    /** Pages of a scan that was never saved (0 = none). */
    unfinishedPages: Int,
    continueUnfinished: () -> Unit,
    discardUnfinished: () -> Unit,
    openDocument: (id: Long) -> Unit,
    openSettings: () -> Unit,
) {
    val vm: HomeViewModel = viewModel { HomeViewModel(app.repository, app.settings) }
    val view by vm.view.collectAsStateWithLifecycle()
    var showImport by remember { mutableStateOf(false) }
    var showSort by remember { mutableStateOf(false) }
    var askUnfinished by remember { mutableStateOf(false) }
    val pickPhotos = rememberLauncherForActivityResult(ActivityResultContracts.PickMultipleVisualMedia(30)) { uris ->
        if (uris.isNotEmpty()) importPhotos(uris)
    }
    val documents by vm.documents.collectAsStateWithLifecycle()
    val query by vm.query.collectAsStateWithLifecycle()
    val selected by vm.selected.collectAsStateWithLifecycle()
    val busy by vm.busy.collectAsStateWithLifecycle()
    val message by vm.message.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }

    var showMerge by remember { mutableStateOf(false) }
    var showDelete by remember { mutableStateOf(false) }
    val selecting = selected.isNotEmpty()

    LaunchedEffect(message) {
        message?.let { snackbar.showSnackbar(it); vm.messageShown() }
    }
    BackHandler(enabled = selecting) { vm.clearSelection() }

    val pickPdf = rememberLauncherForActivityResult(ActivityResultContracts.OpenDocument()) { uri ->
        if (uri != null) {
            vm.importPdf(uri, displayName(context, uri)) { id -> openDocument(id) }
        }
    }

    Scaffold(
        topBar = {
            if (selecting) {
                TopAppBar(
                    title = { Text("${selected.size} selected") },
                    navigationIcon = {
                        IconButton(onClick = { vm.clearSelection() }) { Icon(Icons.Outlined.Close, "Cancel selection") }
                    },
                    actions = {
                        if (selected.size >= 2) {
                            IconButton(onClick = { showMerge = true }) { Icon(Icons.Outlined.MergeType, "Merge into one PDF") }
                        }
                        IconButton(onClick = { showDelete = true }) { Icon(Icons.Outlined.Delete, "Delete") }
                    },
                )
            } else {
                TopAppBar(
                    title = {
                        Column {
                            Text("My Scans", fontWeight = FontWeight.Bold)
                            Text(
                                "Everything stays on this phone",
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    },
                    actions = {
                        Box {
                            IconButton(onClick = { showImport = true }) { Icon(Icons.Outlined.FileOpen, "Import") }
                            DropdownMenu(expanded = showImport, onDismissRequest = { showImport = false }) {
                                DropdownMenuItem(
                                    text = { Text("Import photos") },
                                    leadingIcon = { Icon(Icons.Outlined.PhotoLibrary, null) },
                                    onClick = {
                                        showImport = false
                                        pickPhotos.launch(PickVisualMediaRequest(ActivityResultContracts.PickVisualMedia.ImageOnly))
                                    },
                                )
                                DropdownMenuItem(
                                    text = { Text("Import a PDF") },
                                    leadingIcon = { Icon(Icons.Outlined.PictureAsPdf, null) },
                                    onClick = { showImport = false; pickPdf.launch(arrayOf("application/pdf")) },
                                )
                            }
                        }
                        Box {
                            IconButton(onClick = { showSort = true }) { Icon(Icons.AutoMirrored.Outlined.Sort, "Sort and view") }
                            DropdownMenu(expanded = showSort, onDismissRequest = { showSort = false }) {
                                HomeSort.entries.forEach { sort ->
                                    DropdownMenuItem(
                                        text = { Text(sort.label) },
                                        leadingIcon = { if (view.homeSort == sort) Icon(Icons.Outlined.Check, "Selected") else Spacer(Modifier.width(24.dp)) },
                                        onClick = { showSort = false; vm.setSort(sort) },
                                    )
                                }
                                HorizontalDivider()
                                DropdownMenuItem(
                                    text = { Text(if (view.homeGrid) "Show as list" else "Show as grid") },
                                    leadingIcon = { Icon(if (view.homeGrid) Icons.Outlined.ViewList else Icons.Outlined.GridView, null) },
                                    onClick = { showSort = false; vm.setGrid(!view.homeGrid) },
                                )
                            }
                        }
                        IconButton(onClick = openSettings) { Icon(Icons.Outlined.Settings, "Settings") }
                    },
                )
            }
        },
        floatingActionButton = {
            if (!selecting) {
                ExtendedFloatingActionButton(
                    onClick = { if (unfinishedPages > 0) askUnfinished = true else startScan() },
                    icon = { Icon(Icons.Outlined.DocumentScanner, contentDescription = null) },
                    text = { Text("Scan", style = MaterialTheme.typography.titleMedium) },
                    shape = RoundedCornerShape(20.dp),
                )
            }
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            TextField(
                value = query,
                onValueChange = vm::setQuery,
                placeholder = { Text("Search titles and text") },
                leadingIcon = { Icon(Icons.Outlined.Search, contentDescription = null) },
                singleLine = true,
                shape = RoundedCornerShape(24.dp),
                colors = TextFieldDefaults.colors(
                    focusedIndicatorColor = Color.Transparent,
                    unfocusedIndicatorColor = Color.Transparent,
                    unfocusedContainerColor = MaterialTheme.colorScheme.surfaceContainerLowest,
                    focusedContainerColor = MaterialTheme.colorScheme.surfaceContainerLowest,
                ),
                modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 8.dp),
            )

            if (unfinishedPages > 0) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.tertiaryContainer),
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
                ) {
                    Row(Modifier.padding(start = 16.dp, end = 4.dp, top = 4.dp, bottom = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            if (unfinishedPages == 1) "A scan with 1 page wasn't saved" else "A scan with $unfinishedPages pages wasn't saved",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.weight(1f),
                        )
                        TextButton(onClick = discardUnfinished) { Text("Discard") }
                        TextButton(onClick = continueUnfinished) { Text("Continue", fontWeight = FontWeight.Bold) }
                    }
                }
            }

            val docs = documents
            when {
                docs == null -> Unit
                docs.isEmpty() && query.isBlank() -> EmptyState()
                docs.isEmpty() -> Text(
                    "No scans match “$query”",
                    modifier = Modifier.fillMaxWidth().padding(32.dp),
                    textAlign = TextAlign.Center,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
                view.homeGrid -> LazyVerticalGrid(
                    columns = GridCells.Fixed(2),
                    contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 100.dp),
                    horizontalArrangement = Arrangement.spacedBy(12.dp),
                    verticalArrangement = Arrangement.spacedBy(12.dp),
                ) {
                    gridItems(docs, key = { it.document.id }) { doc ->
                        val id = doc.document.id
                        DocumentTile(
                            doc = doc,
                            selected = id in selected,
                            onClick = { if (selecting) vm.toggleSelected(id) else openDocument(id) },
                            onLongClick = { vm.toggleSelected(id) },
                        )
                    }
                }
                else -> LazyColumn(
                    contentPadding = PaddingValues(start = 16.dp, end = 16.dp, top = 8.dp, bottom = 100.dp),
                    verticalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    items(docs, key = { it.document.id }) { doc ->
                        val id = doc.document.id
                        DocumentRow(
                            doc = doc,
                            selected = id in selected,
                            onClick = { if (selecting) vm.toggleSelected(id) else openDocument(id) },
                            onLongClick = { vm.toggleSelected(id) },
                        )
                    }
                }
            }
        }
    }

    if (askUnfinished) {
        AlertDialog(
            onDismissRequest = { askUnfinished = false },
            title = { Text("You have an unsaved scan") },
            text = { Text("Continue the scan that wasn't saved ($unfinishedPages page(s)), or throw it away and start a new one?") },
            confirmButton = { TextButton(onClick = { askUnfinished = false; continueUnfinished() }) { Text("Continue it") } },
            dismissButton = { TextButton(onClick = { askUnfinished = false; discardUnfinished(); startScan() }) { Text("Start new") } },
        )
    }
    if (showMerge) {
        TextInputDialog(
            title = "Merge ${selected.size} documents",
            initial = "Merged document",
            confirmLabel = "Merge",
            onConfirm = { title ->
                showMerge = false
                vm.mergeSelected(title) { id -> openDocument(id) }
            },
            onDismiss = { showMerge = false },
        )
    }
    if (showDelete) {
        ConfirmDialog(
            title = "Delete ${selected.size} document(s)?",
            message = "The scans will be removed from this phone. This can't be undone.",
            confirmLabel = "Delete",
            onConfirm = { showDelete = false; vm.deleteSelected() },
            onDismiss = { showDelete = false },
        )
    }
    busy?.let { BusyDialog(it) }

    // The app crashed last time: offer to share the details (helps fix the bug).
    var crashReport by remember { mutableStateOf(app.crashFile.takeIf { it.exists() }?.let { runCatching { it.readText() }.getOrNull() }) }
    crashReport?.let { report ->
        AlertDialog(
            onDismissRequest = { app.crashFile.delete(); crashReport = null },
            title = { Text("Sorry, the app closed unexpectedly") },
            text = { Text("You can share the error details with the developer to help fix it. They contain no documents or personal data.") },
            confirmButton = {
                TextButton(onClick = {
                    shareText(context, "My Scanner crash report\n\n$report")
                    app.crashFile.delete(); crashReport = null
                }) { Text("Share report") }
            },
            dismissButton = { TextButton(onClick = { app.crashFile.delete(); crashReport = null }) { Text("Not now") } },
        )
    }
}

@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun DocumentRow(
    doc: DocumentWithPages,
    selected: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
) {
    val pages = doc.pages
    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer
            else MaterialTheme.colorScheme.surfaceContainerLowest,
        ),
        modifier = Modifier
            .fillMaxWidth()
            .combinedClickable(onClick = onClick, onLongClick = onLongClick),
    ) {
        Row(Modifier.padding(12.dp), verticalAlignment = Alignment.CenterVertically) {
            Box(Modifier.size(width = 52.dp, height = 68.dp)) {
                AsyncImage(
                    // The small thumbnail file, not the full page: the list scrolls smoothly.
                    model = pages.firstOrNull()?.let { FileStore.thumbOrImage(it.imagePath) },
                    contentDescription = null,
                    contentScale = ContentScale.Crop,
                    modifier = Modifier.fillMaxSize(),
                )
                if (selected) {
                    Icon(
                        Icons.Filled.CheckCircle,
                        contentDescription = "Selected",
                        tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.align(Alignment.Center).size(28.dp),
                    )
                }
            }
            Column(Modifier.weight(1f).padding(start = 14.dp)) {
                Text(
                    doc.document.title,
                    style = MaterialTheme.typography.titleMedium,
                    maxLines = 1,
                    overflow = TextOverflow.Ellipsis,
                )
                Spacer(Modifier.height(4.dp))
                val count = if (pages.size == 1) "1 page" else "${pages.size} pages"
                Text(
                    "$count · " + SimpleDateFormat("d MMM yyyy", Locale.getDefault()).format(Date(doc.document.updatedAt)),
                    style = MaterialTheme.typography.bodyMedium,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}

/** A document in the grid view: a big thumbnail of its first page with the name under it. */
@OptIn(ExperimentalFoundationApi::class)
@Composable
private fun DocumentTile(
    doc: DocumentWithPages,
    selected: Boolean,
    onClick: () -> Unit,
    onLongClick: () -> Unit,
) {
    val pages = doc.pages
    Card(
        shape = RoundedCornerShape(16.dp),
        colors = CardDefaults.cardColors(
            containerColor = if (selected) MaterialTheme.colorScheme.primaryContainer
            else MaterialTheme.colorScheme.surfaceContainerLowest,
        ),
        modifier = Modifier.fillMaxWidth().combinedClickable(onClick = onClick, onLongClick = onLongClick),
    ) {
        Column(Modifier.padding(8.dp)) {
            Box(Modifier.fillMaxWidth().aspectRatio(0.8f)) {
                AsyncImage(
                    model = pages.firstOrNull()?.let { FileStore.thumbOrImage(it.imagePath) },
                    contentDescription = null,
                    contentScale = ContentScale.Fit,
                    modifier = Modifier.fillMaxSize(),
                )
                if (selected) {
                    Icon(Icons.Filled.CheckCircle, "Selected", tint = MaterialTheme.colorScheme.primary,
                        modifier = Modifier.align(Alignment.Center).size(40.dp))
                }
            }
            Text(doc.document.title, style = MaterialTheme.typography.titleSmall, maxLines = 1, overflow = TextOverflow.Ellipsis,
                modifier = Modifier.padding(top = 6.dp))
            Text(
                (if (pages.size == 1) "1 page" else "${pages.size} pages") + " · " +
                    SimpleDateFormat("d MMM", Locale.getDefault()).format(Date(doc.document.updatedAt)),
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
        }
    }
}

@Composable
private fun EmptyState() {
    Column(
        Modifier.fillMaxSize().padding(32.dp),
        verticalArrangement = Arrangement.Center,
        horizontalAlignment = Alignment.CenterHorizontally,
    ) {
        Icon(
            Icons.Outlined.DocumentScanner,
            contentDescription = null,
            modifier = Modifier.size(64.dp),
            tint = MaterialTheme.colorScheme.primary,
        )
        Spacer(Modifier.height(16.dp))
        Text("No scans yet", style = MaterialTheme.typography.titleLarge)
        Spacer(Modifier.height(8.dp))
        Text(
            "Tap Scan to scan a document. You can scan many pages in one go.",
            textAlign = TextAlign.Center,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
        )
    }
}

/** The file name of a PDF picked from the phone, without ".pdf". */
private fun displayName(context: Context, uri: Uri): String? =
    context.contentResolver.query(uri, arrayOf(OpenableColumns.DISPLAY_NAME), null, null, null)?.use { c ->
        if (c.moveToFirst()) c.getString(0)?.removeSuffix(".pdf")?.removeSuffix(".PDF") else null
    }
