package com.myscanner.app.data

import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import java.io.File
import java.util.UUID

/**
 * Keeps page images in the app's private storage: files/docs/<documentId>/<random>.jpg
 * No storage permission is needed for this folder, and other apps can't read it.
 */
class FileStore(private val context: Context) {

    private fun docDir(documentId: Long): File =
        File(context.filesDir, "docs/$documentId").apply { mkdirs() }

    fun newPageFile(documentId: Long): File =
        File(docDir(documentId), "${UUID.randomUUID()}.jpg")

    /** Copies an image the scanner (or another app) handed us into our own folder. */
    fun copyFromUri(uri: Uri, documentId: Long): File {
        val target = newPageFile(documentId)
        val input = context.contentResolver.openInputStream(uri)
            ?: error("Could not open the scanned image")
        input.use { inp -> target.outputStream().use { out -> inp.copyTo(out) } }
        return target
    }

    fun copyFile(source: File, documentId: Long): File {
        val target = newPageFile(documentId)
        source.copyTo(target)
        return target
    }

    fun saveBitmap(bitmap: Bitmap, documentId: Long, jpegQuality: Int = 90): File {
        val target = newPageFile(documentId)
        target.outputStream().use { bitmap.compress(Bitmap.CompressFormat.JPEG, jpegQuality, it) }
        return target
    }

    /** Moves a photo into the document's folder (as the page's original), returning its new place. */
    fun adoptSource(file: File, documentId: Long): File {
        val target = File(docDir(documentId), "src_${UUID.randomUUID()}.jpg")
        if (!file.renameTo(target)) {
            file.copyTo(target, overwrite = true)
            file.delete()
        }
        return target
    }

    fun deleteFile(path: String) {
        File(path).delete()
        thumbFor(path).delete()
    }

    /** Deletes a page's image, its small thumbnail and its kept original photo. */
    fun deletePageFiles(page: PageEntity) {
        deleteFile(page.imagePath)
        page.sourcePath?.let { File(it).delete() }
    }

    /** Makes the small thumbnail for a page image from an already decoded bitmap (any size). */
    fun saveThumb(imagePath: String, bitmap: Bitmap) {
        runCatching {
            val scale = THUMB_SIDE.toFloat() / maxOf(bitmap.width, bitmap.height)
            val small = if (scale < 1f) Bitmap.createScaledBitmap(
                bitmap, (bitmap.width * scale).toInt().coerceAtLeast(1), (bitmap.height * scale).toInt().coerceAtLeast(1), true,
            ) else bitmap
            thumbFor(imagePath).outputStream().use { small.compress(Bitmap.CompressFormat.JPEG, 82, it) }
            if (small !== bitmap) small.recycle()
        }
    }

    /** Makes the thumbnail by reading the page image at reduced size (for pages that have none yet). */
    fun makeThumb(imagePath: String) {
        runCatching {
            val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
            BitmapFactory.decodeFile(imagePath, bounds)
            var sample = 1
            while (maxOf(bounds.outWidth, bounds.outHeight) / (sample * 2) >= THUMB_SIDE) sample *= 2
            val bmp = BitmapFactory.decodeFile(imagePath, BitmapFactory.Options().apply { inSampleSize = sample }) ?: return
            saveThumb(imagePath, bmp)
            bmp.recycle()
        }
    }

    /** Total size in bytes of all saved scans, and of the kept original photos among them. */
    fun storageUse(): Pair<Long, Long> {
        var all = 0L; var originals = 0L
        File(context.filesDir, "docs").walkTopDown().filter { it.isFile }.forEach { f ->
            all += f.length()
            if (f.name.startsWith("src_")) originals += f.length()
        }
        return all to originals
    }

    /** Deletes every kept original photo (the saved pages themselves stay). Returns the bytes freed. */
    fun deleteAllSources(): Long {
        var freed = 0L
        File(context.filesDir, "docs").walkTopDown().filter { it.isFile && it.name.startsWith("src_") }.toList().forEach { f ->
            freed += f.length(); f.delete()
        }
        return freed
    }

    fun deleteDocumentFolder(documentId: Long) {
        File(context.filesDir, "docs/$documentId").deleteRecursively()
    }

    /**
     * A fresh temporary folder for a PDF or images being saved, shared or printed. Older exports
     * are cleared, except very recent ones (another app may still be reading a file just shared).
     */
    fun exportsDir(): File {
        val root = File(context.cacheDir, "exports")
        val now = System.currentTimeMillis()
        root.listFiles()?.forEach { if (now - it.lastModified() > 10 * 60 * 1000) it.deleteRecursively() }
        return File(root, now.toString()).apply { mkdirs() }
    }

    companion object {
        /** Long side of list thumbnails in pixels. */
        const val THUMB_SIDE = 480

        /** Where the small thumbnail of a page image lives (next to it). */
        fun thumbFor(imagePath: String): File = File(imagePath).let { File(it.parentFile, "thumb_" + it.name) }

        /** The thumbnail if it exists, else the page image itself (older pages get thumbnails in the background). */
        fun thumbOrImage(imagePath: String): File = thumbFor(imagePath).takeIf { it.exists() } ?: File(imagePath)
    }
}
