package com.myscanner.app.data

import android.content.Context
import androidx.datastore.preferences.core.Preferences
import androidx.datastore.preferences.core.booleanPreferencesKey
import androidx.datastore.preferences.core.edit
import androidx.datastore.preferences.core.stringPreferencesKey
import androidx.datastore.preferences.preferencesDataStore
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.map

/** How each scanned page is cleaned up. AUTO removes shadows automatically. */
enum class ScanFilter(val label: String, val description: String) {
    AUTO("Auto", "Removes shadows, white paper, crisp text, natural ink colours"),
    MAGIC("Magic color", "Like Auto but bolder: stronger contrast and colours"),
    LIGHTEN("Lighten", "Brighter and colour-balanced, for photos and colourful pages"),
    GRAY("Grayscale", "Shadow-free grey, smaller files"),
    BW("B&W", "Pure black and white text, smallest files"),
    BW_SOFT("B&W soft", "Black and white with smooth edges, easy to read"),
    ORIGINAL("Original", "Cropped only, colours as photographed"),
}

/** PDF size choices shown when saving. */
enum class PdfQuality(
    val label: String,
    val description: String,
    /** Longest side of each page image in pixels. 0 = keep the original. */
    val maxSide: Int,
    val jpegQuality: Int,
) {
    ORIGINAL("Original", "Best quality, biggest file", 0, 92),
    MEDIUM("Medium", "Sharp and readable, much smaller", 2200, 80),
    SMALL("Small", "Smallest file, good for WhatsApp or email", 1500, 62),
}

/** Camera resolution for scans. Pages are processed and saved at up to [maxSide] pixels. */
enum class PhotoQuality(
    val label: String,
    val description: String,
    /** Camera target in sensor orientation (landscape), 4:3. */
    val captureWidth: Int,
    val captureHeight: Int,
    val maxSide: Int,
) {
    HIGH("High", "Up to 13 MP: sharpest small print, bigger files", 4160, 3120, 4000),
    STANDARD("Standard", "5 MP: faster on budget phones, smaller files", 2592, 1944, 3000),
}

/** Paper size of the PDF pages. */
enum class PdfPageSize(val label: String, val widthPt: Float, val heightPt: Float) {
    /** Each PDF page is exactly as tall as its scan (no white borders). */
    FIT("Fit to scan", 0f, 0f),
    A4("A4", 595f, 842f),
    LETTER("Letter", 612f, 792f),
}

/** Choices made in the Save / Share sheet. */
data class PdfOptions(
    val quality: PdfQuality,
    /** Add an invisible text layer so the PDF can be searched and copied from. */
    val searchable: Boolean = true,
    /** Empty or null = no password. */
    val password: String? = null,
    val pageSize: PdfPageSize = PdfPageSize.FIT,
    /** Print "3 / 12" at the bottom of every page. */
    val pageNumbers: Boolean = false,
    /** Faint text across every page, e.g. "COPY". Empty or null = none. */
    val watermark: String? = null,
)

/** Order of the documents on the home screen. */
enum class HomeSort(val label: String) {
    NEWEST("Newest first"),
    OLDEST("Oldest first"),
    NAME("Name (A to Z)"),
}

enum class ThemeMode(val label: String) {
    SYSTEM("Follow system"),
    LIGHT("Light"),
    DARK("Dark"),
}

data class AppSettings(
    val defaultFilter: ScanFilter = ScanFilter.AUTO,
    val smartNames: Boolean = true,
    val photoQuality: PhotoQuality = PhotoQuality.HIGH,
    val enhanceAnimation: Boolean = true,
    val autoCapture: Boolean = false,
    /** Show the page outline live on the camera picture (off: corners are found after each photo). */
    val liveOutline: Boolean = true,
    val defaultQuality: PdfQuality = PdfQuality.MEDIUM,
    val themeMode: ThemeMode = ThemeMode.SYSTEM,
    /** Keep each page's original photo after saving, so it can be re-cropped later (uses more storage). */
    val keepOriginals: Boolean = true,
    val homeSort: HomeSort = HomeSort.NEWEST,
    /** Home screen as a grid of big thumbnails instead of a list. */
    val homeGrid: Boolean = false,
)

private val Context.dataStore by preferencesDataStore(name = "settings")

class SettingsStore(private val context: Context) {

    private object Keys {
        val defaultFilter = stringPreferencesKey("default_filter")
        val smartNames = booleanPreferencesKey("smart_names")
        val photoQuality = stringPreferencesKey("photo_quality")
        val enhanceAnimation = booleanPreferencesKey("enhance_animation")
        val autoCapture = booleanPreferencesKey("auto_capture")
        val liveOutline = booleanPreferencesKey("live_outline")
        val defaultQuality = stringPreferencesKey("default_quality")
        val themeMode = stringPreferencesKey("theme_mode")
        val keepOriginals = booleanPreferencesKey("keep_originals")
        val homeSort = stringPreferencesKey("home_sort")
        val homeGrid = booleanPreferencesKey("home_grid")
    }

    val settings: Flow<AppSettings> = context.dataStore.data.map { prefs -> prefs.toSettings() }

    private fun Preferences.toSettings() = AppSettings(
        defaultFilter = enumOrDefault(this[Keys.defaultFilter], ScanFilter.AUTO),
        smartNames = this[Keys.smartNames] ?: true,
        photoQuality = enumOrDefault(this[Keys.photoQuality], PhotoQuality.HIGH),
        enhanceAnimation = this[Keys.enhanceAnimation] ?: true,
        autoCapture = this[Keys.autoCapture] ?: false,
        liveOutline = this[Keys.liveOutline] ?: true,
        defaultQuality = enumOrDefault(this[Keys.defaultQuality], PdfQuality.MEDIUM),
        themeMode = enumOrDefault(this[Keys.themeMode], ThemeMode.SYSTEM),
        keepOriginals = this[Keys.keepOriginals] ?: true,
        homeSort = enumOrDefault(this[Keys.homeSort], HomeSort.NEWEST),
        homeGrid = this[Keys.homeGrid] ?: false,
    )

    suspend fun setDefaultFilter(f: ScanFilter) = context.dataStore.edit { it[Keys.defaultFilter] = f.name }
    suspend fun setEnhanceAnimation(on: Boolean) = context.dataStore.edit { it[Keys.enhanceAnimation] = on }
    suspend fun setAutoCapture(on: Boolean) = context.dataStore.edit { it[Keys.autoCapture] = on }
    suspend fun setLiveOutline(on: Boolean) = context.dataStore.edit { it[Keys.liveOutline] = on }
    suspend fun setPhotoQuality(q: PhotoQuality) = context.dataStore.edit { it[Keys.photoQuality] = q.name }
    suspend fun setSmartNames(on: Boolean) = context.dataStore.edit { it[Keys.smartNames] = on }
    suspend fun setDefaultQuality(q: PdfQuality) = context.dataStore.edit { it[Keys.defaultQuality] = q.name }
    suspend fun setThemeMode(mode: ThemeMode) = context.dataStore.edit { it[Keys.themeMode] = mode.name }
    suspend fun setKeepOriginals(on: Boolean) = context.dataStore.edit { it[Keys.keepOriginals] = on }
    suspend fun setHomeSort(sort: HomeSort) = context.dataStore.edit { it[Keys.homeSort] = sort.name }
    suspend fun setHomeGrid(on: Boolean) = context.dataStore.edit { it[Keys.homeGrid] = on }
}

private inline fun <reified T : Enum<T>> enumOrDefault(name: String?, default: T): T =
    enumValues<T>().firstOrNull { it.name == name } ?: default
