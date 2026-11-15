package com.myscanner.app.data

import android.content.Context
import android.graphics.BitmapFactory
import android.net.Uri
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.text.Text
import com.google.mlkit.vision.text.TextRecognition
import com.google.mlkit.vision.text.latin.TextRecognizerOptions
import kotlinx.coroutines.tasks.await
import java.io.File

/**
 * Offline text recognition (OCR) with Google ML Kit.
 * The Latin model is bundled in the app, so it works with no internet from the first launch.
 * It reads English and other Latin-script languages. Bangla needs a different engine (later version).
 */
class TextReader(private val context: Context) {

    private val recognizer by lazy { TextRecognition.getClient(TextRecognizerOptions.DEFAULT_OPTIONS) }

    /** Full result, with each line's position and size (used for smart names). */
    suspend fun readText(imageFile: File): Text {
        // Read a copy at most ~2500 px: plenty for text, and light on memory.
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(imageFile.path, bounds)
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 2500) sample *= 2
        val bmp = BitmapFactory.decodeFile(imageFile.path, BitmapFactory.Options().apply { inSampleSize = sample })
            ?: return recognizer.process(InputImage.fromFilePath(context, Uri.fromFile(imageFile))).await()
        try {
            return recognizer.process(InputImage.fromBitmap(bmp, 0)).await()
        } finally {
            bmp.recycle()
        }
    }

    suspend fun read(imageFile: File): String = readText(imageFile).text

    /** Lines of text with their boxes, for the searchable-PDF layer. */
    suspend fun readLayer(imageFile: File): PdfTools.TextLayer {
        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(imageFile.path, bounds)
        var sample = 1
        while (maxOf(bounds.outWidth, bounds.outHeight) / sample > 2500) sample *= 2
        val bmp = BitmapFactory.decodeFile(imageFile.path, BitmapFactory.Options().apply { inSampleSize = sample })
            ?: return PdfTools.TextLayer(0, 0, emptyList())
        try {
            val result = recognizer.process(InputImage.fromBitmap(bmp, 0)).await()
            val lines = result.textBlocks.flatMap { it.lines }.mapNotNull { l -> l.boundingBox?.let { it to l.text } }
            return PdfTools.TextLayer(bmp.width, bmp.height, lines)
        } finally {
            bmp.recycle()
        }
    }
}
