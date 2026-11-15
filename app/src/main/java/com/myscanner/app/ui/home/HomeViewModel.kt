package com.myscanner.app.ui.home

import android.net.Uri
import androidx.lifecycle.ViewModel
import androidx.lifecycle.viewModelScope
import com.myscanner.app.data.DocumentRepository
import com.myscanner.app.data.DocumentWithPages
import com.myscanner.app.data.AppSettings
import com.myscanner.app.data.HomeSort
import com.myscanner.app.data.SettingsStore
import kotlinx.coroutines.flow.combine
import kotlinx.coroutines.flow.distinctUntilChanged
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.SharingStarted
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.flatMapLatest
import kotlinx.coroutines.flow.stateIn
import kotlinx.coroutines.launch

class HomeViewModel(
    private val repository: DocumentRepository,
    private val settingsStore: SettingsStore,
) : ViewModel() {

    /** How the list is shown: order, and list or grid. */
    val view: StateFlow<AppSettings> = settingsStore.settings
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), AppSettings())

    fun setSort(sort: HomeSort) = viewModelScope.launch { settingsStore.setHomeSort(sort) }
    fun setGrid(on: Boolean) = viewModelScope.launch { settingsStore.setHomeGrid(on) }

    private val _query = MutableStateFlow("")
    val query: StateFlow<String> = _query.asStateFlow()

    @OptIn(ExperimentalCoroutinesApi::class)
    val documents: StateFlow<List<DocumentWithPages>?> = _query
        .flatMapLatest { repository.observeDocuments(it) }
        .combine(settingsStore.settings.map { it.homeSort }.distinctUntilChanged()) { docs, sort ->
            when (sort) {
                HomeSort.NEWEST -> docs.sortedByDescending { it.document.updatedAt }
                HomeSort.OLDEST -> docs.sortedBy { it.document.updatedAt }
                HomeSort.NAME -> docs.sortedBy { it.document.title.lowercase() }
            }
        }
        .stateIn(viewModelScope, SharingStarted.WhileSubscribed(5_000), null)

    /** Documents picked with a long-press, in the order they were picked (used for merging). */
    private val _selected = MutableStateFlow<List<Long>>(emptyList())
    val selected: StateFlow<List<Long>> = _selected.asStateFlow()

    private val _busy = MutableStateFlow<String?>(null)
    val busy: StateFlow<String?> = _busy.asStateFlow()

    private val _message = MutableStateFlow<String?>(null)
    val message: StateFlow<String?> = _message.asStateFlow()

    fun setQuery(text: String) { _query.value = text }

    fun toggleSelected(id: Long) {
        _selected.value = if (id in _selected.value) _selected.value - id else _selected.value + id
    }

    fun clearSelection() { _selected.value = emptyList() }

    fun showMessage(text: String) { _message.value = text }
    fun messageShown() { _message.value = null }

    fun importPdf(uri: Uri, name: String?, onImported: (Long) -> Unit) = viewModelScope.launch {
        _busy.value = "Importing PDF…"
        try {
            onImported(repository.importPdf(uri, name))
        } catch (e: Exception) {
            _message.value = "Could not import: ${e.message}"
        } finally {
            _busy.value = null
        }
    }

    fun mergeSelected(title: String, onMerged: (Long) -> Unit) = viewModelScope.launch {
        val ids = _selected.value
        if (ids.size < 2) return@launch
        _busy.value = "Merging…"
        try {
            val newId = repository.merge(ids, title.trim().ifEmpty { "Merged document" })
            clearSelection()
            onMerged(newId)
        } catch (e: Exception) {
            _message.value = "Could not merge: ${e.message}"
        } finally {
            _busy.value = null
        }
    }

    fun deleteSelected() = viewModelScope.launch {
        val ids = _selected.value
        clearSelection()
        ids.forEach { repository.deleteDocument(it) }
        _message.value = if (ids.size == 1) "Deleted 1 document" else "Deleted ${ids.size} documents"
    }
}
