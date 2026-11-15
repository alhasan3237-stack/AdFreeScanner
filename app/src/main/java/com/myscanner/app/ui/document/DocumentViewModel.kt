package com.myscanner.app.ui.document

import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.myscanner.app.data.AppSettings
import com.myscanner.app.data.DocumentRepository
import com.myscanner.app.data.DocumentWithPages
import com.myscanner.app.data.PdfOptions
import com.myscanner.app.data.PdfQuality
import com.myscanner.app.data.SettingsStore
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch
import java.io.File

class DocumentViewModel(
    private val repository: DocumentRepository,
    settingsStore: SettingsStore,
    private val documentId: Long,
) : ViewModel() {

    val document: StateFlow<DocumentWithPages?> = repository.observeDocument(documentId)
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    val settings: StateFlow<AppSettings> = settingsStore.settings
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), AppSettings())

    private val _estimates = MutableStateFlow<Map<PdfQuality, Long>?>(null)
    val estimates: StateFlow<Map<PdfQuality, Long>?> = _estimates.asStateFlow()

    private val _busy = MutableStateFlow<String?>(null)
    val busy: StateFlow<String?> = _busy.asStateFlow()

    private val _message = MutableStateFlow<String?>(null)
    val message: StateFlow<String?> = _message.asStateFlow()

    fun showMessage(text: String) { _message.value = text }
    fun messageShown() { _message.value = null }

    fun loadEstimates() = viewModelScope.launch {
        _estimates.value = null
        _estimates.value = try { repository.estimateSizes(documentId) } catch (e: Exception) { emptyMap() }
    }

    fun rename(title: String) = viewModelScope.launch { repository.rename(documentId, title) }

    fun delete(onDone: () -> Unit) = viewModelScope.launch {
        repository.deleteDocument(documentId)
        onDone()
    }

    /** Re-reads the first page and renames the document from its text. */
    fun smartRename() = viewModelScope.launch {
        _busy.value = "Reading the first page…"
        try {
            val name = repository.suggestName(documentId)
            if (name != null) {
                repository.rename(documentId, name)
                _message.value = "Renamed to “$name”"
            }
        } catch (e: Exception) {
            _message.value = "Couldn't read the page: ${e.message}"
        } finally {
            _busy.value = null
        }
    }

    fun reorder(pageIds: List<Long>) = viewModelScope.launch { repository.reorderPages(documentId, pageIds) }

    fun savePdf(options: PdfOptions) = viewModelScope.launch {
        _busy.value = if (options.searchable) "Saving PDF (reading the text)…" else "Saving PDF…"
        try {
            val name = repository.savePdfToDownloads(documentId, options)
            _message.value = "Saved to Downloads as $name"
        } catch (e: Exception) {
            _message.value = "Could not save: ${e.message}"
        } finally {
            _busy.value = null
        }
    }

    /** Every page as a picture in the phone's gallery. */
    fun saveImages() = viewModelScope.launch {
        _busy.value = "Saving pictures…"
        try {
            val n = repository.saveImagesToGallery(documentId)
            _message.value = if (n == 1) "Saved 1 picture to Pictures/My Scanner" else "Saved $n pictures to Pictures/My Scanner"
        } catch (e: Exception) {
            _message.value = "Could not save: ${e.message}"
        } finally {
            _busy.value = null
        }
    }

    fun buildImagesForSharing(onReady: (List<File>) -> Unit) = viewModelScope.launch {
        _busy.value = "Preparing pictures…"
        try {
            onReady(repository.exportImages(documentId))
        } catch (e: Exception) {
            _message.value = "Could not prepare the pictures: ${e.message}"
        } finally {
            _busy.value = null
        }
    }

    fun buildPdfForSharing(options: PdfOptions, onReady: (File) -> Unit) = viewModelScope.launch {
        _busy.value = if (options.searchable) "Preparing PDF (reading the text)…" else "Preparing PDF…"
        try {
            onReady(repository.exportPdf(documentId, options))
        } catch (e: Exception) {
            _message.value = "Could not create the PDF: ${e.message}"
        } finally {
            _busy.value = null
        }
    }
}
