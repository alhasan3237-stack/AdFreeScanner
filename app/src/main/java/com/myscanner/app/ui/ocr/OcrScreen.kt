package com.myscanner.app.ui.ocr

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.PaddingValues
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.itemsIndexed
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.text.selection.SelectionContainer
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.ContentCopy
import androidx.compose.material.icons.outlined.Lock
import androidx.compose.material.icons.outlined.Share
import androidx.compose.material3.Button
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedButton
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Scaffold
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableIntStateOf
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalClipboardManager
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.AnnotatedString
import androidx.compose.ui.unit.dp
import androidx.lifecycle.ViewModel
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.lifecycle.viewModelScope
import androidx.lifecycle.viewmodel.compose.viewModel
import com.myscanner.app.ScannerApp
import com.myscanner.app.data.DocumentRepository
import com.myscanner.app.data.DocumentWithPages
import com.myscanner.app.data.pages
import com.myscanner.app.ui.components.shareText
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class OcrViewModel(private val repository: DocumentRepository, documentId: Long) : ViewModel() {

    val document: StateFlow<DocumentWithPages?> = repository.observeDocument(documentId)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    /** (pages done, pages to do) while reading; null when finished. */
    private val _progress = MutableStateFlow<Pair<Int, Int>?>(0 to 0)
    val progress: StateFlow<Pair<Int, Int>?> = _progress.asStateFlow()

    init {
        // Reads any pages that haven't been read yet. Already-read pages are instant.
        viewModelScope.launch {
            try {
                repository.extractText(documentId) { done, total -> _progress.value = done to total }
            } catch (e: Exception) {
                // Nothing to show: the pages that were read are already saved.
            } finally {
                _progress.value = null
            }
        }
    }

    fun saveText(pageId: Long, text: String) = viewModelScope.launch { repository.saveEditedText(pageId, text) }
}

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun OcrScreen(app: ScannerApp, documentId: Long, onBack: () -> Unit) {
    val vm: OcrViewModel = viewModel(key = "ocr-$documentId") { OcrViewModel(app.repository, documentId) }
    val doc by vm.document.collectAsStateWithLifecycle()
    val progress by vm.progress.collectAsStateWithLifecycle()
    val clipboard = LocalClipboardManager.current
    val context = LocalContext.current
    val snackbar = remember { SnackbarHostState() }
    val scope = rememberCoroutineScope()

    // 0 = all pages, 1..n = a single page
    var selectedPage by remember { mutableIntStateOf(0) }
    val pages = doc?.pages.orEmpty()

    val shownText = if (selectedPage == 0) {
        pages.mapIndexedNotNull { i, p ->
            p.ocrText?.takeIf { it.isNotBlank() }?.let { if (pages.size > 1) "— Page ${i + 1} —\n$it" else it }
        }.joinToString("\n\n")
    } else {
        pages.getOrNull(selectedPage - 1)?.ocrText.orEmpty()
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Extracted text") },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Outlined.ArrowBack, "Back") }
                },
            )
        },
        bottomBar = {
            Row(
                Modifier.fillMaxWidth().navigationBarsPadding().padding(16.dp),
                horizontalArrangement = Arrangement.spacedBy(12.dp),
            ) {
                Button(
                    onClick = {
                        clipboard.setText(AnnotatedString(shownText))
                        scope.launch { snackbar.showSnackbar("Copied") }
                    },
                    enabled = shownText.isNotBlank(),
                    modifier = Modifier.weight(1f).height(52.dp),
                ) {
                    Icon(Icons.Outlined.ContentCopy, null)
                    Spacer(Modifier.width(8.dp))
                    Text("Copy")
                }
                OutlinedButton(
                    onClick = { shareText(context, shownText) },
                    enabled = shownText.isNotBlank(),
                    modifier = Modifier.weight(1f).height(52.dp),
                ) {
                    Icon(Icons.Outlined.Share, null)
                    Spacer(Modifier.width(8.dp))
                    Text("Share")
                }
            }
        },
        snackbarHost = { SnackbarHost(snackbar) },
    ) { padding ->
        Column(Modifier.padding(padding).fillMaxSize()) {
            progress?.let { (done, total) ->
                Column(Modifier.fillMaxWidth().padding(horizontal = 20.dp, vertical = 8.dp)) {
                    Text(
                        if (total == 0) "Reading text…" else "Reading page ${minOf(done + 1, total)} of $total…",
                        style = MaterialTheme.typography.bodyMedium,
                    )
                    Spacer(Modifier.height(8.dp))
                    if (total == 0) LinearProgressIndicator(Modifier.fillMaxWidth())
                    else LinearProgressIndicator(progress = { done.toFloat() / total }, modifier = Modifier.fillMaxWidth())
                }
            }

            if (pages.size > 1) {
                LazyRow(
                    contentPadding = PaddingValues(horizontal = 16.dp, vertical = 4.dp),
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                ) {
                    item {
                        FilterChip(selected = selectedPage == 0, onClick = { selectedPage = 0 }, label = { Text("All pages") })
                    }
                    itemsIndexed(pages, key = { _, p -> p.id }) { i, _ ->
                        FilterChip(
                            selected = selectedPage == i + 1,
                            onClick = { selectedPage = i + 1 },
                            label = { Text("Page ${i + 1}") },
                        )
                    }
                }
            }

            if (selectedPage == 0) {
                Card(
                    colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLowest),
                    modifier = Modifier.weight(1f).fillMaxWidth().padding(16.dp),
                ) {
                    SelectionContainer {
                        Text(
                            shownText.ifBlank { if (progress == null) "No text found in this document." else "" },
                            style = MaterialTheme.typography.bodyLarge,
                            modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(20.dp),
                        )
                    }
                }
            } else {
                val page = pages.getOrNull(selectedPage - 1)
                if (page != null) {
                    // Single page: the text can be corrected by hand and saved.
                    var edited by remember(page.id, page.ocrText) { mutableStateOf(page.ocrText.orEmpty()) }
                    OutlinedTextField(
                        value = edited,
                        onValueChange = { edited = it },
                        placeholder = { Text("No text found on this page") },
                        modifier = Modifier.weight(1f).fillMaxWidth().padding(16.dp),
                    )
                    if (edited != page.ocrText.orEmpty()) {
                        TextButton(
                            onClick = { vm.saveText(page.id, edited) },
                            modifier = Modifier.align(Alignment.End).padding(end = 12.dp),
                        ) { Text("Save changes") }
                    }
                }
            }

            Row(
                Modifier.padding(horizontal = 20.dp),
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Icon(Icons.Outlined.Lock, null, Modifier.size(14.dp), tint = MaterialTheme.colorScheme.onSurfaceVariant)
                Spacer(Modifier.width(6.dp))
                Text(
                    "Read on this phone · English and other Latin-script languages",
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
    }
}
