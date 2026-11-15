package com.myscanner.app.scan

import android.content.Context
import android.graphics.Bitmap
import android.util.LruCache
import com.myscanner.app.data.PhotoQuality
import com.myscanner.app.data.ScanFilter
import kotlinx.coroutines.CoroutineDispatcher
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.ExperimentalCoroutinesApi
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.update
import org.opencv.core.Mat
import org.opencv.core.Size
import org.opencv.imgproc.Imgproc
import java.io.File
import java.util.Base64
import java.util.UUID
import java.util.concurrent.atomic.AtomicLong
import kotlin.math.max

/** One photographed page waiting to be saved. */
data class ScanPage(
    val id: Long,
    /** The full photo, as taken. Never changed. */
    val file: File,
    val quad: Quad,
    val filter: ScanFilter,
    /** Extra clockwise rotation chosen by the user: 0, 90, 180, 270. */
    val rotation: Int = 0,
    /** False when the detector wasn't sure of the outline (the user should check the corners). */
    val detected: Boolean = true,
    /** Pen strokes, highlighter, signature and typed text drawn on the page. */
    val marks: List<Mark> = emptyList(),
)

/**
 * The pages of one scanning session (camera → review → save).
 * Photos live in a temporary folder until the user saves. The list of pages is also written to
 * that folder after every change, so an unfinished scan survives the app being closed by the
 * system (common on phones with little memory) and can be continued.
 */
class ScanSession(
    context: Context,
    val targetDocumentId: Long?,
    val defaultFilter: ScanFilter,
    val photoQuality: PhotoQuality,
    /** Editing a saved document: saving replaces its pages instead of adding new ones. */
    val replaceDocumentId: Long? = null,
    /** Only given when a session is brought back from disk. */
    existingFolder: File? = null,
) {
    val isEditing: Boolean get() = replaceDocumentId != null

    /** Biggest side of a saved page, in pixels. */
    val outputMaxSide: Int get() = photoQuality.maxSide

    val folder: File = (existingFolder ?: File(root(context), UUID.randomUUID().toString())).apply { mkdirs() }

    private val nextId = AtomicLong(1)
    private val _pages = MutableStateFlow<List<ScanPage>>(emptyList())
    val pages: StateFlow<List<ScanPage>> = _pages.asStateFlow()

    /** How many photos are still being checked for page edges. */
    private val _processing = MutableStateFlow(0)
    val processing: StateFlow<Int> = _processing.asStateFlow()

    /** Which page the review screen should open on (null = the newest). Used when editing. */
    var openAtPageIndex: Int? = null

    fun newPhotoFile(): File = File(folder, "${UUID.randomUUID()}.jpg")

    fun startedProcessing() = _processing.update { it + 1 }
    fun finishedProcessing() = _processing.update { (it - 1).coerceAtLeast(0) }

    /** Adds a page and returns its id. [detected] = the edge detector was sure of the outline. */
    fun add(file: File, quad: Quad, detected: Boolean): Long {
        val page = ScanPage(nextId.getAndIncrement(), file, quad, defaultFilter, 0, detected)
        _pages.update { it + page }
        persist()
        return page.id
    }

    /** Adds a page that already has corners, filter, rotation and drawings (editing a saved page). */
    fun addExisting(file: File, quad: Quad, filter: ScanFilter, rotation: Int, marks: List<Mark> = emptyList(), detected: Boolean = true): Long {
        val page = ScanPage(nextId.getAndIncrement(), file, quad, filter, rotation, detected, marks)
        _pages.update { it + page }
        persist()
        return page.id
    }

    fun page(id: Long): ScanPage? = _pages.value.firstOrNull { it.id == id }

    fun update(id: Long, change: (ScanPage) -> ScanPage) {
        _pages.update { list -> list.map { if (it.id == id) change(it) else it } }
        persist()
    }

    /** Turns a page clockwise by 90°, taking its drawings along. */
    fun rotate(id: Long) = update(id) { it.copy(rotation = (it.rotation + 90) % 360, marks = Marks.rotate(it.marks, 90)) }

    fun setFilterForAll(filter: ScanFilter) {
        _pages.update { list -> list.map { it.copy(filter = filter) } }
        persist()
    }

    /** Moves a page one place earlier (-1) or later (+1) in the scan. */
    fun move(id: Long, by: Int) {
        _pages.update { list ->
            val i = list.indexOfFirst { it.id == id }
            val j = i + by
            if (i < 0 || j !in list.indices) list else list.toMutableList().apply { add(j, removeAt(i)) }
        }
        persist()
    }

    /** Replaces two pages with one combined page (e.g. an ID card sheet), at the first one's place. */
    fun replaceWithCombined(firstId: Long, secondId: Long, file: File): Long {
        val newPage = ScanPage(nextId.getAndIncrement(), file, Quad.WHOLE, ScanFilter.ORIGINAL, 0, true)
        val old = _pages.value.filter { it.id == firstId || it.id == secondId }
        _pages.update { list ->
            val i = list.indexOfFirst { it.id == firstId }
            val rest = list.filterNot { it.id == firstId || it.id == secondId }.toMutableList()
            rest.add(i.coerceIn(0, rest.size), newPage)
            rest
        }
        old.forEach { it.file.delete() }
        persist()
        return newPage.id
    }

    fun remove(id: Long) {
        val removed = page(id)
        _pages.update { list -> list.filterNot { it.id == id } }
        removed?.file?.delete()
        persist()
    }

    fun discard() {
        _pages.value = emptyList()
        folder.deleteRecursively()
    }

    // ---------- keeping the session on disk ----------

    private fun persist() {
        runCatching {
            if (!folder.exists()) return
            val sb = StringBuilder("v1\n")
            sb.append("target=").append(targetDocumentId ?: "").append('\n')
            sb.append("replace=").append(replaceDocumentId ?: "").append('\n')
            sb.append("filter=").append(defaultFilter.name).append('\n')
            sb.append("quality=").append(photoQuality.name).append('\n')
            for (p in _pages.value) {
                sb.append("page\t").append(p.file.name).append('\t').append(p.quad.encode()).append('\t')
                    .append(p.filter.name).append('\t').append(p.rotation).append('\t').append(if (p.detected) 1 else 0).append('\t')
                    .append(Base64.getEncoder().encodeToString(Marks.encode(p.marks).toByteArray(Charsets.UTF_8)))
                    .append('\t').append(p.id).append('\n')
            }
            // Write to a spare file first, then swap: a crash mid-write can't leave a broken list.
            val tmp = File(folder, "$STATE.tmp")
            tmp.writeText(sb.toString())
            val state = File(folder, STATE)
            if (!tmp.renameTo(state)) { state.delete(); tmp.renameTo(state) }
        }
    }

    companion object {
        private const val STATE = "session.txt"

        fun root(context: Context): File = File(context.cacheDir, "scan-session")

        /**
         * Brings back the newest unfinished session left on disk (or null), and deletes the others.
         * Pages whose photo is missing are skipped.
         */
        fun restore(context: Context): ScanSession? {
            val folders = root(context).listFiles { f -> f.isDirectory }.orEmpty()
            val newest = folders.filter { File(it, STATE).exists() }.maxByOrNull { File(it, STATE).lastModified() }
            val session = newest?.let { runCatching { read(context, it) }.getOrNull() }
            folders.filter { it != session?.folder }.forEach { it.deleteRecursively() }
            return session
        }

        private fun read(context: Context, folder: File): ScanSession? {
            val lines = File(folder, STATE).readLines()
            if (lines.firstOrNull() != "v1") return null
            fun value(key: String) = lines.firstOrNull { it.startsWith("$key=") }?.substringAfter("=").orEmpty()
            val filter = runCatching { ScanFilter.valueOf(value("filter")) }.getOrDefault(ScanFilter.AUTO)
            val quality = runCatching { PhotoQuality.valueOf(value("quality")) }.getOrDefault(PhotoQuality.HIGH)
            val session = ScanSession(
                context, value("target").toLongOrNull(), filter, quality,
                replaceDocumentId = value("replace").toLongOrNull(), existingFolder = folder,
            )
            for (line in lines) {
                if (!line.startsWith("page\t")) continue
                val f = line.split("\t")
                val file = File(folder, f.getOrNull(1) ?: continue)
                if (!file.exists()) continue
                val quad = Quad.decode(f.getOrNull(2)) ?: Quad.FULL
                val pageFilter = runCatching { ScanFilter.valueOf(f[3]) }.getOrDefault(filter)
                val marks = runCatching { Marks.decode(String(Base64.getDecoder().decode(f[6]), Charsets.UTF_8)) }.getOrDefault(emptyList())
                // Pages keep their ids, so a screen that was open on "page 7" still finds it.
                val id = f.getOrNull(7)?.toLongOrNull()?.takeIf { wanted -> session._pages.value.none { it.id == wanted } }
                    ?: ((session._pages.value.maxOfOrNull { it.id } ?: 0L) + 1)
                val page = ScanPage(id, file, quad, pageFilter, f.getOrNull(4)?.toIntOrNull() ?: 0, f.getOrNull(5) != "0", marks)
                session._pages.update { it + page }
            }
            session.nextId.set((session._pages.value.maxOfOrNull { it.id } ?: 0L) + 1)
            return session.takeIf { it._pages.value.isNotEmpty() }
        }
    }
}

/**
 * Makes preview images quickly and keeps recent ones in memory, so flipping between pages and
 * filters is instant after the first time. Limited to 2 threads so the phone stays responsive.
 *
 * The slow parts (decoding the photo, straightening, removing shadows) are done once per page and
 * kept ("prepared"); filters, rotation, drawings and the small filter thumbnails are quick steps
 * on top of that.
 */
object PreviewRenderer {

    @OptIn(ExperimentalCoroutinesApi::class)
    val dispatcher: CoroutineDispatcher = Dispatchers.Default.limitedParallelism(2)

    /** Long side of the page previews in pixels. Set once from the screen size (720p phones need less). */
    @Volatile var previewSide: Int = 1600

    /** Long side of the quick, small renders (the capture animation, thumbnails). */
    const val SMALL_SIDE = 720

    private val cache = object : LruCache<String, Bitmap>(
        // Up to 1/6 of the app's memory, in KB.
        (Runtime.getRuntime().maxMemory() / 1024 / 6).toInt()
    ) {
        override fun sizeOf(key: String, value: Bitmap) = value.byteCount / 1024
    }

    /** The straightened page and its shadow-free version, for one page outline at one size. */
    private class Prepared(val page: Mat, val flat: Mat) {
        val kb: Int get() = ((page.total() * page.elemSize() + flat.total() * flat.elemSize()) / 1024).toInt().coerceAtLeast(1)
        fun release() { page.release(); flat.release() }
    }

    /** About 48 MB of prepared pages: 3-4 at preview size, or many small ones. */
    private val prepared = object : LruCache<String, Prepared>(48 * 1024) {
        override fun sizeOf(key: String, value: Prepared) = value.kb
        override fun entryRemoved(evicted: Boolean, key: String, old: Prepared, new: Prepared?) = old.release()
    }

    /** Only one page is prepared at a time: a second request for the same page waits, then finds it ready. */
    private val prepWork = Any()

    private fun prepKey(page: ScanPage, big: Boolean) = "${page.id}|${page.quad.key()}|${if (big) "P" else "S"}"

    /** Copies of the prepared images if they are in memory (copies, so an eviction elsewhere can't free them mid-use). */
    private fun cachedCopies(page: ScanPage, big: Boolean): Pair<Mat, Mat>? = synchronized(prepared) {
        prepared.get(prepKey(page, big))?.let { return it.page.clone() to it.flat.clone() }
        if (!big) {
            // The preview-size version exists: shrink it instead of decoding the photo again.
            prepared.get(prepKey(page, true))?.let { p ->
                val a = Mat(); val b = Mat()
                val s = SMALL_SIDE.toDouble() / max(p.page.cols(), p.page.rows())
                if (s < 1) {
                    val size = Size((p.page.cols() * s).toInt().coerceAtLeast(1).toDouble(), (p.page.rows() * s).toInt().coerceAtLeast(1).toDouble())
                    Imgproc.resize(p.page, a, size, 0.0, 0.0, Imgproc.INTER_AREA)
                    Imgproc.resize(p.flat, b, size, 0.0, 0.0, Imgproc.INTER_AREA)
                } else { p.page.copyTo(a); p.flat.copyTo(b) }
                prepared.put(prepKey(page, false), Prepared(a.clone(), b.clone()))
                return a to b
            }
        }
        null
    }

    private fun preparedCopies(page: ScanPage, big: Boolean): Pair<Mat, Mat> {
        cachedCopies(page, big)?.let { return it }
        synchronized(prepWork) {
            cachedCopies(page, big)?.let { return it }
            val warped = ImageProcessor.warpPage(page.file, page.quad, if (big) previewSide else SMALL_SIDE, fast = true)
            val flat = ImageProcessor.flatten(warped)
            synchronized(prepared) { prepared.put(prepKey(page, big), Prepared(warped.clone(), flat.clone())) }
            return warped to flat
        }
    }

    /**
     * The page as it will look, at most [maxSide] px. Must be called off the main thread (use
     * [dispatcher]). [withMarks] = false leaves the drawings out (the drawing screen shows them itself).
     */
    fun render(page: ScanPage, maxSide: Int, filter: ScanFilter = page.filter, withMarks: Boolean = true): Bitmap {
        val marks = if (withMarks) page.marks else emptyList()
        val key = "${page.id}|${page.quad.key()}|$filter|${page.rotation}|$maxSide|${marks.hashCode()}"
        cache.get(key)?.let { return it }

        val (warped, flat) = preparedCopies(page, big = maxSide > SMALL_SIDE)
        val longest = max(warped.cols(), warped.rows())
        if (maxSide in 1 until longest) {
            // Thumbnails: shrink first, so the filter only works on a few pixels.
            val s = maxSide.toDouble() / longest
            val size = Size((warped.cols() * s).toInt().coerceAtLeast(1).toDouble(), (warped.rows() * s).toInt().coerceAtLeast(1).toDouble())
            Imgproc.resize(warped, warped, size, 0.0, 0.0, Imgproc.INTER_AREA)
            Imgproc.resize(flat, flat, size, 0.0, 0.0, Imgproc.INTER_AREA)
        }
        var mat = ImageProcessor.applyFilter(warped, filter, flat)
        warped.release(); flat.release()
        ImageProcessor.rotateInPlace(mat, page.rotation)
        mat = ImageProcessor.withMarks(mat, marks)
        val bmp = ImageProcessor.toBitmap(mat)
        mat.release()
        cache.put(key, bmp)
        return bmp
    }

    /** True when [render] for this page at preview size would return at once. */
    fun isReady(page: ScanPage): Boolean =
        cache.get("${page.id}|${page.quad.key()}|${page.filter}|${page.rotation}|$previewSide|${page.marks.hashCode()}") != null

    fun clear() {
        cache.evictAll()
        synchronized(prepared) { prepared.evictAll() }
        ImageProcessor.clearCache()
    }
}
