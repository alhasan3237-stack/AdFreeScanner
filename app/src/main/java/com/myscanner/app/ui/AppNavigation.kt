package com.myscanner.app.ui

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.MaterialTheme
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.lifecycle.Lifecycle
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import androidx.navigation.NavBackStackEntry
import androidx.navigation.NavHostController
import androidx.navigation.NavType
import androidx.navigation.compose.NavHost
import androidx.navigation.compose.composable
import androidx.navigation.compose.rememberNavController
import androidx.navigation.navArgument
import com.myscanner.app.ScannerApp
import com.myscanner.app.data.AppSettings
import com.myscanner.app.ui.components.BusyDialog
import com.myscanner.app.ui.document.DocumentScreen
import com.myscanner.app.ui.home.HomeScreen
import com.myscanner.app.ui.ocr.OcrScreen
import com.myscanner.app.ui.page.PageScreen
import com.myscanner.app.ui.scan.CameraScreen
import com.myscanner.app.ui.scan.CropScreen
import com.myscanner.app.ui.scan.MarkupScreen
import android.net.Uri
import com.myscanner.app.scan.ImageProcessor
import com.myscanner.app.scan.importGalleryImage
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.withContext
import com.myscanner.app.ui.scan.ReviewScreen
import com.myscanner.app.ui.settings.SettingsScreen
import kotlinx.coroutines.launch

private const val DOCUMENT_ROUTE = "document/{id}?scanned={scanned}"

/**
 * Goes back one screen, safely: ignored while a screen is still animating in or out (a quick
 * double tap used to go back twice), and never removes the very first screen (that left an
 * empty, white app).
 */
private fun NavHostController.safeBack() {
    val current = currentBackStackEntry ?: return
    if (previousBackStackEntry == null) return
    if (!current.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) return
    popBackStack()
}

/** True when [entry] is the screen on top (not one that is animating away). */
private fun NavHostController.isOnTop(entry: NavBackStackEntry) = currentBackStackEntry?.id == entry.id

/** All the screens and how to move between them. */
@Composable
fun AppNavigation(app: ScannerApp) {
    val nav = rememberNavController()
    val settings by app.settings.settings.collectAsStateWithLifecycle(initialValue = AppSettings())
    val scope = rememberCoroutineScope()
    /** A busy message while pages are being opened or imported; null otherwise. */
    var opening by remember { mutableStateOf<String?>(null) }

    /** Opens the camera. [documentId] = add pages to that document. */
    fun startScan(documentId: Long?) {
        app.startSession(documentId, settings.defaultFilter, settings.photoQuality)
        nav.navigate("camera")
    }

    /** Opens a saved document's pages in the review screen for editing, starting at [openPageId]. */
    fun startEdit(documentId: Long, openPageId: Long? = null) {
        if (opening != null) return
        opening = "Opening the pages for editing…"
        scope.launch {
            val session = app.startEditSession(documentId, settings.defaultFilter, settings.photoQuality, openPageId)
            opening = null
            if (session != null) nav.navigate("review")
        }
    }

    /** Photos picked from the gallery become a new scan: edges are found, then the review screen opens. */
    fun importPhotos(uris: List<Uri>) {
        if (uris.isEmpty() || opening != null) return
        opening = if (uris.size == 1) "Importing the photo…" else "Importing ${uris.size} photos…"
        val session = app.startSession(null, settings.defaultFilter, settings.photoQuality)
        scope.launch {
            withContext(Dispatchers.IO) {
                for (uri in uris) runCatching {
                    val file = session.newPhotoFile()
                    importGalleryImage(app, uri, file, session.outputMaxSide)
                    val (q, sure) = ImageProcessor.detect(file)
                    session.add(file, q, sure)
                }
            }
            opening = null
            if (session.pages.value.isNotEmpty()) nav.navigate("review") else app.endSession()
        }
    }

    /** Leaves the scanning screens, back to where scanning (or editing) started. */
    fun closeScan() {
        val editing = app.session?.isEditing == true
        app.endSession()
        if (editing) {
            nav.popBackStack("review", inclusive = true)
        } else if (!nav.popBackStack("camera", inclusive = true)) {
            nav.popBackStack("review", inclusive = true)
        }
    }

    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background)) {
        NavHost(navController = nav, startDestination = "home") {

            composable("home") {
                // A scan that was never saved (the phone closed the app, or it was swiped away).
                var unfinished by remember { mutableStateOf(app.session?.pages?.value?.size ?: 0) }
                HomeScreen(
                    app = app,
                    startScan = { startScan(null) },
                    importPhotos = { importPhotos(it) },
                    unfinishedPages = unfinished,
                    continueUnfinished = { if (app.session != null) nav.navigate("review") else unfinished = 0 },
                    discardUnfinished = { app.endSession(); unfinished = 0 },
                    openDocument = { id -> nav.navigate("document/$id?scanned=false") },
                    openSettings = { nav.navigate("settings") },
                )
            }

            composable("camera") { entry ->
                val session = app.session
                if (session == null) {
                    SessionGone(nav, entry)
                } else {
                    CameraScreen(
                        session = session,
                        // Back to the review screen if it's already open (adding pages), else open it.
                        onDone = { if (!nav.popBackStack("review", inclusive = false)) nav.navigate("review") },
                        onClose = {
                            if (session.isEditing) nav.popBackStack("review", inclusive = false) else closeScan()
                        },
                        animate = settings.enhanceAnimation,
                        autoCaptureDefault = settings.autoCapture,
                        liveOutline = settings.liveOutline,
                        editing = session.isEditing,
                    )
                }
            }

            composable("review") { entry ->
                val session = app.session
                if (session == null) {
                    SessionGone(nav, entry)
                } else {
                    ReviewScreen(
                        app = app,
                        session = session,
                        onAddMore = {
                            // The camera is under this screen when scanning; when editing, open it.
                            if (nav.previousBackStackEntry?.destination?.route == "camera") nav.safeBack()
                            else nav.navigate("camera")
                        },
                        onCrop = { pageId -> nav.navigate("crop/$pageId") },
                        onMarkup = { pageId -> nav.navigate("markup/$pageId") },
                        onSaved = { docId, wasNew ->
                            if (wasNew) {
                                nav.navigate("document/$docId?scanned=true") {
                                    popUpTo("home") { inclusive = false }
                                }
                            } else if (!nav.popBackStack(DOCUMENT_ROUTE, inclusive = false)) {
                                // Pages were added to (or edited in) a document: go back to it.
                                nav.navigate("document/$docId?scanned=false") { popUpTo("home") { inclusive = false } }
                            }
                        },
                        onDiscard = { closeScan() },
                    )
                }
            }

            composable(
                "crop/{pageId}",
                arguments = listOf(navArgument("pageId") { type = NavType.LongType }),
            ) { entry ->
                val session = app.session
                if (session == null) {
                    SessionGone(nav, entry)
                } else {
                    CropScreen(
                        session = session,
                        pageId = entry.arguments?.getLong("pageId") ?: 0L,
                        onDone = { nav.safeBack() },
                    )
                }
            }

            composable(
                "markup/{pageId}",
                arguments = listOf(navArgument("pageId") { type = NavType.LongType }),
            ) { entry ->
                val session = app.session
                if (session == null) {
                    SessionGone(nav, entry)
                } else {
                    MarkupScreen(
                        session = session,
                        pageId = entry.arguments?.getLong("pageId") ?: 0L,
                        onDone = { nav.safeBack() },
                    )
                }
            }

            composable(
                DOCUMENT_ROUTE,
                arguments = listOf(
                    navArgument("id") { type = NavType.LongType },
                    navArgument("scanned") { type = NavType.BoolType; defaultValue = false },
                ),
            ) { entry ->
                val id = entry.arguments?.getLong("id") ?: 0L
                DocumentScreen(
                    app = app,
                    documentId = id,
                    justScanned = entry.arguments?.getBoolean("scanned") ?: false,
                    onBack = { nav.safeBack() },
                    addPages = { startScan(id) },
                    editPages = { startEdit(id) },
                    openPage = { pageId -> nav.navigate("page/$pageId") },
                    openText = { docId -> nav.navigate("ocr/$docId") },
                )
            }

            composable(
                "page/{pageId}",
                arguments = listOf(navArgument("pageId") { type = NavType.LongType }),
            ) { entry ->
                PageScreen(
                    app = app,
                    pageId = entry.arguments?.getLong("pageId") ?: 0L,
                    onBack = { nav.safeBack() },
                    onDocumentDeleted = { nav.popBackStack("home", inclusive = false) },
                    onEdit = { docId, openPageId -> startEdit(docId, openPageId) },
                )
            }

            composable(
                "ocr/{docId}",
                arguments = listOf(navArgument("docId") { type = NavType.LongType }),
            ) { entry ->
                OcrScreen(
                    app = app,
                    documentId = entry.arguments?.getLong("docId") ?: 0L,
                    onBack = { nav.safeBack() },
                )
            }

            composable("settings") {
                SettingsScreen(app = app, onBack = { nav.safeBack() })
            }
        }
    }
    opening?.let { BusyDialog(it) }
}

/**
 * Shown for a moment by a scanning screen whose session has already ended (after saving or
 * discarding). Goes back only if it is really the screen on top, so it can't pop extra screens.
 */
@Composable
private fun SessionGone(nav: NavHostController, entry: NavBackStackEntry) {
    Box(Modifier.fillMaxSize().background(MaterialTheme.colorScheme.background))
    LaunchedEffect(Unit) {
        // Wait until it has finished appearing (at most ~3 s), then step back once.
        repeat(30) {
            if (!nav.isOnTop(entry)) return@LaunchedEffect
            if (entry.lifecycle.currentState.isAtLeast(Lifecycle.State.RESUMED)) { nav.safeBack(); return@LaunchedEffect }
            kotlinx.coroutines.delay(100)
        }
    }
}
