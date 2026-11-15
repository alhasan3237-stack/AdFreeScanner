package com.myscanner.app.ui.document

import android.Manifest
import android.content.pm.PackageManager
import android.os.Build
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.animation.core.animateDpAsState
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.aspectRatio
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.grid.GridCells
import androidx.compose.foundation.lazy.grid.GridItemSpan
import androidx.compose.foundation.lazy.grid.LazyVerticalGrid
import androidx.compose.foundation.lazy.grid.items
import androidx.compose.foundation.lazy.grid.rememberLazyGridState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.Add
import androidx.compose.material.icons.outlined.AutoAwesome
import androidx.compose.material.icons.outlined.Delete
import androidx.compose.material.icons.outlined.Download
import androidx.compose.material.icons.outlined.DriveFileRenameOutline
import androidx.compose.material.icons.outlined.Edit
import androidx.compose.material.icons.outlined.MoreVert
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material.icons.outlined.TextFields
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.layout.ContentScale
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextOverflow
import androidx.compose.ui.unit.dp
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewmodel.compose.viewModel
import coil.compose.AsyncImage
import com.myscanner.app.ScannerApp
import com.myscanner.app.data.PageEntity
import com.myscanner.app.data.PdfOptions
import com.myscanner.app.data.PdfQuality
import com.myscanner.app.data.pages
import com.myscanner.app.ui.components.BusyDialog
import com.myscanner.app.ui.components.ConfirmDialog
import com.myscanner.app.ui.components.SavePdfSheet
import com.myscanner.app.ui.components.SheetAction
import com.myscanner.app.ui.components.TextInputDialog
import com.myscanner.app.ui.components.sharePdf
import com.myscanner.app.ui.components.shareImages
import com.myscanner.app.ui.components.printPdf
import com.myscanner.app.data.FileStore
import androidx.compose.material.icons.outlined.Image
import androidx.compose.material.icons.outlined.Print
import androidx.compose.material.icons.outlined.Collections
import androidx.compose.material3.HorizontalDivider
import sh.calvin.reorderable.ReorderableItem
import sh.calvin.reorderable.rememberReorderableLazyGridState
import java.io.File

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun DocumentScreen(
    app: ScannerApp,
    documentId: Long,
    justScanned: Boolean,
    onBack: () -> Unit,
    addPages: () -> Unit,
    /** Opens all pages for editing: crop, filters, rotate, reorder, delete, add. */
    editPages: () -> Unit,
    openPage: (pageId: Long) -> Unit,
    openText: (documentId: Long) -> Unit,
) {
    val vm: DocumentViewModel = viewModel(key = "doc-$documentId") {
        DocumentViewModel(app.repository, app.settings, documentId)
    }
    val doc by vm.document.collectAsStateWithLifecycle()
    val settings by vm.settings.collectAsStateWithLifecycle()
    val estimates by vm.estimates.collectAsStateWithLifecycle()
    val busy by vm.busy.collectAsStateWithLifecycle()
    val message by vm.message.collectAsStateWithLifecycle()
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }

    var sheet by remember { mutableStateOf<SheetAction?>(null) }
    var showRename by remember { mutableStateOf(false) }
    var showDelete by remember { mutableStateOf(false) }
    var showMenu by remember { mutableStateOf(false) }
    var showScanBanner by remember { mutableStateOf(justScanned) }
    /** What to do once the storage permission is granted (Android 8-9 only). */
    var pendingSave by remember { mutableStateOf<(() -> Unit)?>(null) }

    LaunchedEffect(message) {
        message?.let { snackbar.showSnackbar(it); vm.messageShown() }
    }

    val scanMore = addPages

    // Android 8-9 only: ask for storage permission before writing into Downloads or Pictures.
    val storagePermission = rememberLauncherForActivityResult(ActivityResultContracts.RequestPermission()) { granted ->
        val action = pendingSave
        pendingSave = null
        if (granted) action?.invoke() else vm.showMessage("Storage permission is needed to save to the phone")
    }
    fun withStorage(action: () -> Unit) {
        val needsPermission = Build.VERSION.SDK_INT < Build.VERSION_CODES.Q &&
            ContextCompat.checkSelfPermission(context, Manifest.permission.WRITE_EXTERNAL_STORAGE) !=
            PackageManager.PERMISSION_GRANTED
        if (needsPermission) {
            pendingSave = action
            storagePermission.launch(Manifest.permission.WRITE_EXTERNAL_STORAGE)
        } else {
            action()
        }
    }
    fun save(options: PdfOptions) = withStorage { vm.savePdf(options) }

    val current = doc
    Scaffold(
        topBar = {
            TopAppBar(
                title = {
                    Text(
                        current?.document?.title ?: "",
                        maxLines = 1,
                        overflow = TextOverflow.Ellipsis,
                        modifier = Modifier.clickable { showRename = true },
                    )
                },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Outlined.ArrowBack, "Back") }
                },
                actions = {
                    Box {
                        IconButton(onClick = { showMenu = true }) { Icon(Icons.Outlined.MoreVert, "More options") }
                        DropdownMenu(expanded = showMenu, onDismissRequest = { showMenu = false }) {
                            DropdownMenuItem(
                                text = { Text("Add pages") },
                                leadingIcon = { Icon(Icons.Outlined.Add, null) },
                                onClick = { showMenu = false; scanMore() },
                            )
                            DropdownMenuItem(
                                text = { Text("Edit pages") },
                                leadingIcon = { Icon(Icons.Outlined.Edit, null) },
                                onClick = { showMenu = false; editPages() },
                            )
                            HorizontalDivider()
                            DropdownMenuItem(
                                text = { Text("Save pages as pictures") },
                                leadingIcon = { Icon(Icons.Outlined.Image, null) },
                                onClick = { showMenu = false; withStorage { vm.saveImages() } },
                            )
                            DropdownMenuItem(
                                text = { Text("Share as pictures") },
                                leadingIcon = { Icon(Icons.Outlined.Collections, null) },
                                onClick = { showMenu = false; vm.buildImagesForSharing { files -> shareImages(context, files) } },
                            )
                            DropdownMenuItem(
                                text = { Text("Print") },
                                leadingIcon = { Icon(Icons.Outlined.Print, null) },
                                onClick = {
                                    showMenu = false
                                    val title = current?.document?.title ?: "Scan"
                                    vm.buildPdfForSharing(PdfOptions(PdfQuality.ORIGINAL, searchable = false)) { file ->
                                        if (!printPdf(context, file, title)) vm.showMessage("Printing isn't available on this phone")
                                    }
                                },
                            )
                            HorizontalDivider()
                            DropdownMenuItem(
                                text = { Text("Rename") },
                                leadingIcon = { Icon(Icons.Outlined.DriveFileRenameOutline, null) },
                                onClick = { showMenu = false; showRename = true },
                            )
                            DropdownMenuItem(
                                text = { Text("Smart name") },
                                leadingIcon = { Icon(Icons.Outlined.AutoAwesome, null) },
                                onClick = { showMenu = false; vm.smartRename() },
                            )
                            DropdownMenuItem(
                                text = { Text("Extract text") },
                                leadingIcon = { Icon(Icons.Outlined.TextFields, null) },
                                onClick = { showMenu = false; openText(documentId) },
                            )
                            DropdownMenuItem(
                                text = { Text("Delete document", color = MaterialTheme.colorScheme.error) },
                                leadingIcon = { Icon(Icons.Outlined.Delete, null, tint = MaterialTheme.colorScheme.error) },
                                onClick = { showMenu = false; showDelete = true },
                            )
                        }
                    }
                },
            )
        },
        bottomBar = {
            Surface(tonalElevation = 3.dp, shadowElevation = 8.dp) {
                Row(
                    Modifier.fillMaxWidth().navigationBarsPadding().padding(horizontal = 16.dp, vertical = 12.dp),
                    horizontalArrangement = Arrangement.spacedBy(10.dp),
                ) {
                    OutlinedButton(
                        onClick = editPages,
                        modifier = Modifier.height(52.dp),
                    ) {
                        Icon(Icons.Outlined.Edit, null)
                        Spacer(Modifier.width(6.dp))
                        Text("Edit")
                    }
                    OutlinedButton(
                        onClick = { sheet = SheetAction.SHARE; vm.loadEstimates() },
                        modifier = Modifier.height(52.dp),
                    ) {
                        Icon(Icons.Outlined.Share, "Share")
                    }
                    OutlinedButton(
                        onClick = { openText(documentId) },
                        modifier = Modifier.height(52.dp),
                    ) {
                        Icon(Icons.Outlined.TextFields, "Extract text")
                    }
                    Button(
                        onClick = { sheet = SheetAction.SAVE; vm.loadEstimates() },
                        modifier = Modifier.weight(1f).height(52.dp),
                    ) {
                        Icon(Icons.Outlined.Download, null)
                        Spacer(Modifier.width(6.dp))
                        Text("Save PDF", style = MaterialTheme.typography.titleMedium)
                    }
                }
            }
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        if (current == null) return@Scaffold
        Column(Modifier.padding(padding).fillMaxSize()) {
            if (showScanBanner) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.primaryContainer),
                    modifier = Modifier.fillMaxWidth().padding(horizontal = 16.dp, vertical = 4.dp),
                ) {
                    Row(Modifier.padding(start = 16.dp, top = 6.dp, bottom = 6.dp, end = 4.dp), verticalAlignment = Alignment.CenterVertically) {
                        Text(
                            "Scan saved as “${current.document.title}”. Tap the name to change it, or Save PDF for a copy in Downloads.",
                            style = MaterialTheme.typography.bodyMedium,
                            modifier = Modifier.weight(1f),
                        )
                        androidx.compose.material3.TextButton(onClick = { showScanBanner = false }) { Text("OK") }
                    }
                }
            }
            val count = current.pages.size
            Text(
                (if (count == 1) "1 page" else "$count pages") + " · press and hold a page to drag it",
                style = MaterialTheme.typography.bodySmall,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
                modifier = Modifier.padding(horizontal = 20.dp, vertical = 8.dp),
            )
            PageGrid(
                pages = current.pages,
                onOpen = openPage,
                onReorder = vm::reorder,
                onAddPages = scanMore,
            )
        }
    }

    sheet?.let { action ->
        SavePdfSheet(
            action = action,
            defaultQuality = settings.defaultQuality,
            estimates = estimates,
            onConfirm = { quality: PdfOptions ->
                sheet = null
                showScanBanner = false
                when (action) {
                    SheetAction.SAVE -> save(quality)
                    SheetAction.SHARE -> vm.buildPdfForSharing(quality) { file -> sharePdf(context, file) }
                }
            },
            onDismiss = { sheet = null },
        )
    }
    if (showRename && current != null) {
        TextInputDialog(
            title = "Rename",
            initial = current.document.title,
            confirmLabel = "Save",
            onConfirm = { showRename = false; vm.rename(it) },
            onDismiss = { showRename = false },
        )
    }
    if (showDelete) {
        ConfirmDialog(
            title = "Delete this document?",
            message = "All its pages will be removed from this phone. This can't be undone.",
            confirmLabel = "Delete",
            onConfirm = { showDelete = false; vm.delete(onBack) },
            onDismiss = { showDelete = false },
        )
    }
    busy?.let { BusyDialog(it) }
}

/** Grid of page thumbnails. Press and hold a page to drag it to a new place. */
@Composable
private fun PageGrid(
    pages: List<PageEntity>,
    onOpen: (Long) -> Unit,
    onReorder: (List<Long>) -> Unit,
    onAddPages: () -> Unit,
) {
    var localPages by remember { mutableStateOf(pages) }
    var dragging by remember { mutableStateOf(false) }
    LaunchedEffect(pages) { if (!dragging) localPages = pages }

    val gridState = rememberLazyGridState()
    val reorderState = rememberReorderableLazyGridState(gridState) { from, to ->
        val fromIndex = localPages.indexOfFirst { it.id == from.key }
        val toIndex = localPages.indexOfFirst { it.id == to.key }
        if (fromIndex >= 0 && toIndex >= 0) {
            localPages = localPages.toMutableList().apply { add(toIndex, removeAt(fromIndex)) }
        }
    }

    LazyVerticalGrid(
        columns = GridCells.Fixed(2),
        state = gridState,
        contentPadding = PaddingValues(start = 16.dp, end = 16.dp, bottom = 24.dp),
        horizontalArrangement = Arrangement.spacedBy(14.dp),
        verticalArrangement = Arrangement.spacedBy(14.dp),
        modifier = Modifier.fillMaxSize(),
    ) {
        items(localPages, key = { it.id }) { page ->
            ReorderableItem(reorderState, key = page.id) { isDragging ->
                val elevation by animateDpAsState(if (isDragging) 10.dp else 1.dp, label = "drag")
                val number = localPages.indexOf(page) + 1
                Column(horizontalAlignment = Alignment.CenterHorizontally) {
                    Card(
                        elevation = CardDefaults.cardElevation(defaultElevation = elevation),
                        shape = RoundedCornerShape(10.dp),
                        colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLowest),
                        modifier = Modifier
                            .fillMaxWidth()
                            .aspectRatio(0.72f)
                            .longPressDraggableHandle(
                                onDragStarted = { dragging = true },
                                onDragStopped = {
                                    dragging = false
                                    onReorder(localPages.map { it.id })
                                },
                            )
                            .clickable { onOpen(page.id) },
                    ) {
                        AsyncImage(
                            // The small thumbnail file: the grid scrolls smoothly even with many pages.
                            model = FileStore.thumbOrImage(page.imagePath),
                            contentDescription = "Page $number",
                            contentScale = ContentScale.Fit,
                            modifier = Modifier.fillMaxSize().padding(4.dp),
                        )
                    }
                    Text(
                        "Page $number",
                        style = MaterialTheme.typography.labelMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        fontWeight = FontWeight.Medium,
                        modifier = Modifier.padding(top = 6.dp),
                    )
                }
            }
        }
        item(span = { GridItemSpan(maxLineSpan) }) {
            OutlinedButton(
                onClick = onAddPages,
                border = BorderStroke(1.5.dp, MaterialTheme.colorScheme.outline),
                shape = RoundedCornerShape(14.dp),
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) {
                Icon(Icons.Outlined.Add, null)
                Spacer(Modifier.width(8.dp))
                Text("Add pages")
            }
        }
    }
}
