package com.myscanner.app.data

import android.content.Context
import android.net.Uri
import com.myscanner.app.scan.HeavyWork
import com.myscanner.app.scan.ImageProcessor
import com.myscanner.app.scan.Marks
import com.myscanner.app.scan.Quad
import com.myscanner.app.scan.ScanSession
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import org.opencv.core.Mat
import org.opencv.imgproc.Imgproc
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * The single place screens go to read or change documents.
 * It keeps the database and the image files in step with each other.
 */
class DocumentRepository(
    private val context: Context,
    private val dao: DocumentDao,
    private val files: FileStore,
    /** Lives as long as the app: used for work that should finish even if you leave the screen. */
    private val appScope: CoroutineScope,
) {
    private val textReader = TextReader(context)

    /** One text-reading job at a time (two at once would read the same pages twice). */
    private val ocrLock = Mutex()

    fun observeDocuments(query: String): Flow<List<DocumentWithPages>> =
        if (query.isBlank()) dao.observeAll() else dao.search(query.trim())

    fun observeDocument(id: Long): Flow<DocumentWithPages?> = dao.observeDocument(id)

    suspend fun getDocument(id: Long): DocumentWithPages? = dao.getDocument(id)

    suspend fun getPage(id: Long): PageEntity? = dao.getPage(id)

    private fun now() = System.currentTimeMillis()

    private fun defaultTitle(prefix: String = "Scan"): String =
        prefix + " " + SimpleDateFormat("d MMM yyyy, HH:mm", Locale.getDefault()).format(Date())

    /** Writes the small list thumbnail for a page image from the finished page (any size). */
    private fun saveThumb(page: Mat, imageFile: File) {
        runCatching {
            val small = Mat()
            val s = FileStore.THUMB_SIDE.toDouble() / maxOf(page.cols(), page.rows())
            if (s < 1) Imgproc.resize(page, small, org.opencv.core.Size(page.cols() * s, page.rows() * s), 0.0, 0.0, Imgproc.INTER_AREA)
            else page.copyTo(small)
            ImageProcessor.saveJpeg(small, FileStore.thumbFor(imageFile.path), 82)
            small.release()
        }
    }

    /**
     * Saves the pages of a scanning session: straightens and cleans each page at full quality,
     * then creates the document (or adds to / replaces the pages of an existing one).
     * With [smartNames], a new document is named from the text on its first page.
     * With [keepOriginals], each page's original photo is kept so it can be re-cropped later.
     * Returns the document id.
     */
    suspend fun saveScanSession(
        session: ScanSession,
        smartNames: Boolean,
        keepOriginals: Boolean,
        onProgress: (done: Int, total: Int) -> Unit,
    ): Long = withContext(Dispatchers.IO) {
        val pages = session.pages.value
        require(pages.isNotEmpty()) { "Nothing to save" }
        val replacing = session.replaceDocumentId?.let { dao.getDocument(it) }
        val existing = replacing ?: session.targetDocumentId?.let { dao.getDocument(it) }
        val docId = existing?.document?.id
            ?: dao.insertDocument(DocumentEntity(title = defaultTitle(), createdAt = now(), updatedAt = now()))

        val saved = mutableListOf<File>()
        try {
            pages.forEachIndexed { i, page ->
                onProgress(i, pages.size)
                val out = files.newPageFile(docId)
                HeavyWork.lock.withLock {
                    val mat = ImageProcessor.render(page.file, page.quad, page.filter, page.rotation, session.outputMaxSide, page.marks)
                    try {
                        ImageProcessor.saveJpeg(mat, out)
                        saveThumb(mat, out)
                    } finally {
                        mat.release()
                    }
                }
                saved += out
            }
        } catch (e: Throwable) {
            saved.forEach { files.deleteFile(it.path) }
            if (existing == null) deleteDocument(docId)
            throw e
        }
        onProgress(pages.size, pages.size)

        // Keep each original photo with its corners/filter/rotation/drawings, so the page can be edited later.
        val sources = pages.map { page -> if (keepOriginals) runCatching { files.adoptSource(page.file, docId) }.getOrNull() else null }
        fun entity(i: Int, position: Int) = PageEntity(
            documentId = docId,
            position = position,
            imagePath = saved[i].path,
            sourcePath = sources[i]?.path,
            quad = pages[i].quad.encode(),
            filter = pages[i].filter.name,
            rotation = pages[i].rotation,
            // Without the original the drawings are already part of the saved picture.
            marks = if (sources[i] != null) Marks.encode(pages[i].marks).takeIf { it.isNotEmpty() } else null,
        )
        if (replacing != null) {
            // Editing: the edited pages replace the old ones (same document, same name).
            val old = replacing.pagesUnsorted
            dao.deletePagesOf(docId)
            dao.insertPages(saved.indices.map { entity(it, it) })
            old.forEach { files.deletePageFiles(it) }
        } else {
            val start = existing?.pagesUnsorted?.size ?: 0
            dao.insertPages(saved.indices.map { entity(it, start + it) })
        }
        dao.touch(docId, now())
        session.discard()

        if (existing == null && smartNames) nameFromFirstPage(docId)

        // Read the rest of the text in the background, so search works without opening "Extract text".
        appScope.launch { runCatching { extractText(docId) } }
        docId
    }

    /**
     * Copies a saved document's pages into a new editing session: original photos with their
     * corners, filter, rotation and drawings when kept, otherwise the saved page itself (whole image).
     * [openPageId] = the page the review screen should open on.
     */
    suspend fun loadForEditing(documentId: Long, session: ScanSession, openPageId: Long? = null): Boolean = withContext(Dispatchers.IO) {
        val doc = dao.getDocument(documentId) ?: return@withContext false
        val pages = doc.pages
        for (p in pages) {
            val source = p.sourcePath?.let { File(it) }?.takeIf { it.exists() }
            val copy = session.newPhotoFile()
            if (source != null) {
                source.copyTo(copy, overwrite = true)
                val filter = runCatching { ScanFilter.valueOf(p.filter ?: "") }.getOrDefault(ScanFilter.ORIGINAL)
                session.addExisting(copy, Quad.decode(p.quad) ?: Quad.WHOLE, filter, p.rotation, Marks.decode(p.marks))
            } else {
                File(p.imagePath).copyTo(copy, overwrite = true)
                session.addExisting(copy, Quad.WHOLE, ScanFilter.ORIGINAL, 0)
            }
        }
        session.openAtPageIndex = pages.indexOfFirst { it.id == openPageId }.takeIf { it >= 0 } ?: 0
        pages.isNotEmpty()
    }

    /** Reads the first page and names the document from it. Keeps the default name if nothing useful is found. */
    private suspend fun nameFromFirstPage(docId: Long) {
        val doc = dao.getDocument(docId) ?: return
        val first = doc.pages.firstOrNull() ?: return
        val text = try {
            textReader.readText(File(first.imagePath))
        } catch (e: Exception) {
            null
        }
        if (text != null) dao.setOcrText(first.id, text.text)
        dao.rename(docId, SmartNamer.suggest(text, doc.document.createdAt), now())
    }

    /** Suggests a name for an existing document from its first page (the "Smart name" button). */
    suspend fun suggestName(docId: Long): String? = withContext(Dispatchers.IO) {
        val doc = dao.getDocument(docId) ?: return@withContext null
        val first = doc.pages.firstOrNull() ?: return@withContext null
        val text = try { textReader.readText(File(first.imagePath)) } catch (e: Exception) { null }
        SmartNamer.suggest(text, doc.document.createdAt)
    }

    /** Turns a PDF from the phone into a document (each page becomes an image). */
    suspend fun importPdf(uri: Uri, title: String?): Long = withContext(Dispatchers.IO) {
        val id = dao.insertDocument(
            DocumentEntity(title = title ?: defaultTitle("Imported"), createdAt = now(), updatedAt = now())
        )
        try {
            val saved = mutableListOf<File>()
            PdfTools.renderPdfPages(context, uri) { bitmap ->
                val f = files.saveBitmap(bitmap, id)
                files.saveThumb(f.path, bitmap)
                saved += f
            }
            dao.insertPages(saved.mapIndexed { i, f -> PageEntity(documentId = id, position = i, imagePath = f.path) })
            id
        } catch (e: Exception) {
            dao.deleteDocument(id)
            files.deleteDocumentFolder(id)
            throw e
        }
    }

    /** Joins several documents, in the order given, into one new document. */
    suspend fun merge(documentIds: List<Long>, title: String): Long = withContext(Dispatchers.IO) {
        val id = dao.insertDocument(DocumentEntity(title = title, createdAt = now(), updatedAt = now()))
        var position = 0
        val pages = mutableListOf<PageEntity>()
        for (sourceId in documentIds) {
            val source = dao.getDocument(sourceId) ?: continue
            for (page in source.pages) {
                val copy = files.copyFile(File(page.imagePath), id)
                files.makeThumb(copy.path)
                pages += PageEntity(documentId = id, position = position++, imagePath = copy.path, ocrText = page.ocrText)
            }
        }
        dao.insertPages(pages)
        id
    }

    suspend fun rename(documentId: Long, title: String) =
        dao.rename(documentId, title.trim().ifEmpty { "Untitled" }, now())

    suspend fun deleteDocument(documentId: Long) = withContext(Dispatchers.IO) {
        dao.deleteDocument(documentId)
        files.deleteDocumentFolder(documentId)
    }

    /** Deletes one page and closes the gap in the page order. Deletes the document if it was the last page. */
    suspend fun deletePage(pageId: Long): Boolean = withContext(Dispatchers.IO) {
        val page = dao.getPage(pageId) ?: return@withContext false
        dao.deletePage(pageId)
        files.deletePageFiles(page)
        val remaining = dao.getDocument(page.documentId)?.pages.orEmpty()
        if (remaining.isEmpty()) {
            deleteDocument(page.documentId)
            true
        } else {
            dao.setPositions(remaining.map { it.id })
            dao.touch(page.documentId, now())
            false
        }
    }

    /** Saves a new page order after the user drags pages around. */
    suspend fun reorderPages(documentId: Long, orderedPageIds: List<Long>) = withContext(Dispatchers.IO) {
        dao.setPositions(orderedPageIds)
        dao.touch(documentId, now())
    }

    /** Rotates a page image by 90° steps. Saves it as a new file so thumbnails refresh. */
    suspend fun rotatePage(pageId: Long, degrees: Int) = withContext(Dispatchers.IO) {
        val page = dao.getPage(pageId) ?: return@withContext
        val newFile = files.newPageFile(page.documentId)
        HeavyWork.lock.withLock { ImageProcessor.rotateFile(File(page.imagePath), degrees, newFile) }
        files.makeThumb(newFile.path)
        // Remember the rotation (and turn the drawings with it), so editing the page later keeps it the right way up.
        val rotation = ((page.rotation + degrees) % 360 + 360) % 360
        val marks = page.marks?.let { Marks.encode(Marks.rotate(Marks.decode(it), degrees)) }
        dao.setImage(pageId, newFile.path, rotation, marks)
        files.deleteFile(page.imagePath)
        dao.touch(page.documentId, now())
    }

    /** Reads the text on every page that hasn't been read yet. Calls onProgress(done, total). */
    suspend fun extractText(documentId: Long, onProgress: (Int, Int) -> Unit = { _, _ -> }) =
        withContext(Dispatchers.IO) {
            ocrLock.withLock {
                val todo = dao.getDocument(documentId)?.pages.orEmpty().filter { it.ocrText == null }
                todo.forEachIndexed { i, stale ->
                    onProgress(i, todo.size)
                    // The page may have been rotated, replaced or deleted since the list was made.
                    val page = dao.getPage(stale.id)
                    if (page != null && page.ocrText == null) {
                        val text = try {
                            textReader.read(File(page.imagePath))
                        } catch (e: Exception) {
                            ""
                        }
                        dao.setOcrText(page.id, text)
                    }
                }
                onProgress(todo.size, todo.size)
            }
        }

    suspend fun saveEditedText(pageId: Long, text: String) = withContext(Dispatchers.IO) {
        dao.setOcrText(pageId, text)
    }

    /** Size estimates for each quality, shown in the Save sheet. */
    suspend fun estimateSizes(documentId: Long): Map<PdfQuality, Long> = withContext(Dispatchers.Default) {
        val pageFiles = dao.getDocument(documentId)?.pages.orEmpty().map { File(it.imagePath) }
        PdfQuality.entries.associateWith { PdfTools.estimateBytes(pageFiles, it) }
    }

    /** Builds the PDF in a temporary folder and returns the file. */
    suspend fun exportPdf(documentId: Long, options: PdfOptions): File = withContext(Dispatchers.IO) {
        val doc = dao.getDocument(documentId) ?: error("Document not found")
        val pageFiles = doc.pages.map { File(it.imagePath) }
        // Searchable PDF: read each page's text (offline) to place it invisibly over the image.
        val layers = if (options.searchable) pageFiles.map { f -> runCatching { textReader.readLayer(f) }.getOrNull() } else null
        val out = File(files.exportsDir(), PdfTools.fileNameFor(doc.document.title))
        PdfTools.buildPdf(pageFiles, options, out, layers)
        out
    }

    /** One tap: builds the PDF and puts it in Downloads. Returns the saved file name. */
    suspend fun savePdfToDownloads(documentId: Long, options: PdfOptions): String {
        val pdf = exportPdf(documentId, options)
        return withContext(Dispatchers.IO) { PdfTools.saveToDownloads(context, pdf, pdf.name) }
    }

    /** Copies of the page images with readable names ("Title - page 2.jpg"), for sharing as pictures. */
    suspend fun exportImages(documentId: Long): List<File> = withContext(Dispatchers.IO) {
        val doc = dao.getDocument(documentId) ?: error("Document not found")
        val dir = files.exportsDir()
        val base = PdfTools.baseNameFor(doc.document.title)
        val pages = doc.pages
        pages.mapIndexed { i, p ->
            val name = if (pages.size == 1) "$base.jpg" else "$base - page ${i + 1}.jpg"
            File(dir, name).also { File(p.imagePath).copyTo(it, overwrite = true) }
        }
    }

    /** Saves every page as a picture in the phone's gallery (Pictures/My Scanner). Returns how many. */
    suspend fun saveImagesToGallery(documentId: Long): Int = withContext(Dispatchers.IO) {
        val images = exportImages(documentId)
        images.forEach { PdfTools.saveToGallery(context, it, it.name) }
        images.size
    }

    /** Makes list thumbnails for pages saved by older versions of the app (runs once in the background). */
    suspend fun ensureThumbnails() = withContext(Dispatchers.IO) {
        for (p in dao.allPages()) {
            if (!FileStore.thumbFor(p.imagePath).exists() && File(p.imagePath).exists()) files.makeThumb(p.imagePath)
        }
    }

    /** (all scans, kept original photos) in bytes. */
    suspend fun storageUse(): Pair<Long, Long> = withContext(Dispatchers.IO) { files.storageUse() }

    /** Deletes every kept original photo to free space. The saved pages stay; they just can't be re-cropped. */
    suspend fun deleteOriginals(): Long = withContext(Dispatchers.IO) {
        dao.forgetSources()
        files.deleteAllSources()
    }
}
