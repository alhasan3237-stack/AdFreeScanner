package com.myscanner.app

import android.app.Application
import android.util.Log
import com.myscanner.app.data.AppDatabase
import com.myscanner.app.data.DocumentRepository
import com.myscanner.app.data.FileStore
import com.myscanner.app.data.PhotoQuality
import com.myscanner.app.data.ScanFilter
import com.myscanner.app.data.SettingsStore
import com.myscanner.app.scan.PreviewRenderer
import com.myscanner.app.scan.ScanSession
import kotlinx.coroutines.launch
import com.tom_roush.pdfbox.android.PDFBoxResourceLoader
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import org.opencv.android.OpenCVLoader
import java.io.File

/**
 * Runs once when the app starts. Creates the shared objects that every screen uses.
 */
class ScannerApp : Application() {

    lateinit var repository: DocumentRepository
        private set
    lateinit var settings: SettingsStore
        private set

    /** Background work that should keep going even if you leave a screen. */
    val appScope = CoroutineScope(SupervisorJob() + Dispatchers.Default)

    /** The pages being scanned right now (camera → review → save). */
    var session: ScanSession? = null
        private set

    /** Where the last crash's details are kept until the user shares or dismisses them. */
    val crashFile: File get() = File(filesDir, "last_crash.txt")

    override fun onCreate() {
        super.onCreate()
        // If the app ever crashes, keep the details so the user can send them to the developer.
        val previous = Thread.getDefaultUncaughtExceptionHandler()
        Thread.setDefaultUncaughtExceptionHandler { thread, error ->
            runCatching {
                val trace = android.util.Log.getStackTraceString(error)
                crashFile.writeText(
                    "App version: ${packageManager.getPackageInfo(packageName, 0).versionName}\n" +
                        "Phone: ${android.os.Build.MANUFACTURER} ${android.os.Build.MODEL}, Android ${android.os.Build.VERSION.RELEASE}\n" +
                        "Memory: ${Runtime.getRuntime().maxMemory() / 1048576} MB app limit\n\n" + trace
                )
            }
            previous?.uncaughtException(thread, error)
        }
        if (!OpenCVLoader.initLocal()) Log.e("ScannerApp", "OpenCV failed to load")
        com.myscanner.app.scan.NativeGuard.init(this)
        PDFBoxResourceLoader.init(applicationContext)
        val db = AppDatabase.create(this)
        settings = SettingsStore(this)
        repository = DocumentRepository(this, db.documentDao(), FileStore(this), appScope)

        // 720p screens don't need 1600 px previews: smaller ones are made (and shown) faster.
        val screen = resources.displayMetrics
        PreviewRenderer.previewSide = (maxOf(screen.widthPixels, screen.heightPixels) * 0.75).toInt().coerceIn(1000, 1600)

        // A scan that was never saved or discarded (the phone closed the app in the background,
        // or it was swiped away): bring its pages back instead of losing them.
        session = runCatching { ScanSession.restore(this) }.getOrNull()

        // Slow start-up work happens off the main thread, so the first screen appears at once.
        appScope.launch(Dispatchers.IO) {
            com.myscanner.app.scan.PageNet.init(this@ScannerApp)
            com.myscanner.app.scan.DocQuad.init(this@ScannerApp)
            runCatching { repository.ensureThumbnails() }
        }
    }

    /** The system is short of memory: let go of cached images. */
    override fun onTrimMemory(level: Int) {
        super.onTrimMemory(level)
        if (level >= TRIM_MEMORY_RUNNING_LOW) PreviewRenderer.clear()
    }

    /** Starts a fresh scan. [documentId] = add the pages to that document; null = a new document. */
    fun startSession(documentId: Long?, defaultFilter: ScanFilter, photoQuality: PhotoQuality): ScanSession {
        session?.discard()
        PreviewRenderer.clear()
        return ScanSession(this, documentId, defaultFilter, photoQuality).also { session = it }
    }

    /**
     * Opens a saved document for editing: its pages go into a session; saving replaces them.
     * [openPageId] = the page to show first.
     */
    suspend fun startEditSession(documentId: Long, defaultFilter: ScanFilter, photoQuality: PhotoQuality, openPageId: Long? = null): ScanSession? {
        session?.discard()
        PreviewRenderer.clear()
        val s = ScanSession(this, null, defaultFilter, photoQuality, replaceDocumentId = documentId)
        session = s
        val ok = runCatching { repository.loadForEditing(documentId, s, openPageId) }.getOrDefault(false)
        if (!ok) { endSession(); return null }
        return s
    }

    fun endSession() {
        session?.discard()
        session = null
        PreviewRenderer.clear()
    }
}
