package com.myscanner.app.data

import android.content.ContentValues
import android.content.Context
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.graphics.Color
import android.graphics.pdf.PdfRenderer
import android.net.Uri
import android.os.Build
import android.os.Environment
import android.provider.MediaStore
import com.tom_roush.pdfbox.pdmodel.PDDocument
import com.tom_roush.pdfbox.pdmodel.PDPage
import com.tom_roush.pdfbox.pdmodel.PDPageContentStream
import com.tom_roush.pdfbox.pdmodel.common.PDRectangle
import com.tom_roush.pdfbox.pdmodel.encryption.AccessPermission
import com.tom_roush.pdfbox.pdmodel.encryption.StandardProtectionPolicy
import com.tom_roush.pdfbox.pdmodel.font.PDType1Font
import com.tom_roush.pdfbox.pdmodel.graphics.state.PDExtendedGraphicsState
import com.tom_roush.pdfbox.pdmodel.graphics.state.RenderingMode
import com.tom_roush.pdfbox.util.Matrix
import com.tom_roush.pdfbox.pdmodel.graphics.image.JPEGFactory
import java.io.ByteArrayOutputStream
import java.io.File
import kotlin.math.max
import kotlin.math.roundToInt

/** Building, sizing, importing and saving PDFs. Everything here runs offline. */
object PdfTools {

    /** A4 width in PDF points. Each page is this wide; its height follows the image. */
    private const val PAGE_WIDTH = 595f

    /**
     * Builds a PDF from page images. Smaller qualities shrink each image and re-compress it
     * as JPEG, which is what makes the file smaller. [options] also choose the paper size, page
     * numbers, a watermark and a password.
     */
    fun buildPdf(
        pageFiles: List<File>,
        options: PdfOptions,
        out: File,
        textLayers: List<TextLayer?>? = null,
    ) {
        PDDocument().use { doc ->
            val watermark = options.watermark?.let(::latin)?.takeIf { it.isNotEmpty() }
            pageFiles.forEachIndexed { index, file ->
                val bytes = imageBytes(file, options.quality)
                val image = JPEGFactory.createFromByteArray(doc, bytes)
                val iw = image.width.toFloat(); val ih = image.height.toFloat()
                val size = options.pageSize
                // Paper: as tall as the scan, or a fixed size turned to match a landscape scan.
                val pw: Float; val ph: Float
                if (size == PdfPageSize.FIT) { pw = PAGE_WIDTH; ph = PAGE_WIDTH * ih / iw }
                else if (iw > ih) { pw = size.heightPt; ph = size.widthPt }
                else { pw = size.widthPt; ph = size.heightPt }
                val margin = if (size == PdfPageSize.FIT) 0f else 18f
                val scale = minOf((pw - 2 * margin) / iw, (ph - 2 * margin) / ih)
                val dw = iw * scale; val dh = ih * scale
                val x = (pw - dw) / 2; val y = (ph - dh) / 2
                val page = PDPage(PDRectangle(pw, ph))
                doc.addPage(page)
                PDPageContentStream(doc, page).use { stream ->
                    stream.drawImage(image, x, y, dw, dh)
                    textLayers?.getOrNull(index)?.let { addInvisibleText(stream, it, x, y, dw, dh) }
                    if (watermark != null) runCatching { addWatermark(stream, watermark, pw, ph) }
                    if (options.pageNumbers) runCatching { addPageNumber(stream, "${index + 1} / ${pageFiles.size}", pw) }
                }
            }
            val password = options.password
            if (!password.isNullOrEmpty()) {
                // Needs the password to open. AES-128, readable by every PDF app.
                val policy = StandardProtectionPolicy(password, password, AccessPermission())
                policy.encryptionKeyLength = 128
                doc.protect(policy)
            }
            doc.save(out)
        }
    }

    /** The built-in PDF font only knows basic Latin letters: everything else becomes a space. */
    private fun latin(text: String): String = text.map { c -> if (c.code in 32..126) c else ' ' }.joinToString("").trim()

    /** Text found on one page, with each line's box in the coordinates of an image [imageWidth] wide. */
    class TextLayer(val imageWidth: Int, val imageHeight: Int, val lines: List<Pair<android.graphics.Rect, String>>)

    /**
     * Writes the recognised text invisibly on top of the page image (which sits at [x],[y] and is
     * [dw] x [dh] points), so the PDF can be searched and its text selected and copied in any PDF
     * viewer (a "searchable PDF").
     */
    private fun addInvisibleText(stream: PDPageContentStream, layer: TextLayer, x: Float, y: Float, dw: Float, dh: Float) {
        if (layer.imageWidth <= 0) return
        val scale = dw / layer.imageWidth
        val font = PDType1Font.HELVETICA
        stream.setRenderingMode(RenderingMode.NEITHER)
        for ((box, raw) in layer.lines) {
            val text = latin(raw)
            if (text.isEmpty() || box.width() <= 0 || box.height() <= 0) continue
            val fontSize = (box.height() * scale * 0.85f).coerceIn(2f, 200f)
            val natural = runCatching { font.getStringWidth(text) / 1000f * fontSize }.getOrDefault(0f)
            if (natural <= 0f) continue
            val stretch = (box.width() * scale / natural * 100f).coerceIn(10f, 1000f)
            runCatching {
                stream.beginText()
                stream.setFont(font, fontSize)
                stream.setHorizontalScaling(stretch)
                stream.newLineAtOffset(x + box.left * scale, y + dh - box.bottom * scale + fontSize * 0.2f)
                stream.showText(text)
                stream.endText()
            }
        }
        stream.setRenderingMode(RenderingMode.FILL)
        stream.setHorizontalScaling(100f)
    }

    /** Faint grey text running diagonally across the page, twice. */
    private fun addWatermark(stream: PDPageContentStream, text: String, pw: Float, ph: Float) {
        val font = PDType1Font.HELVETICA_BOLD
        val diagonal = kotlin.math.hypot(pw, ph)
        val angle = kotlin.math.atan2(ph, pw)
        // As big as fits along 70% of the diagonal.
        val unit = font.getStringWidth(text) / 1000f
        val fontSize = (diagonal * 0.7f / unit.coerceAtLeast(0.1f)).coerceIn(14f, 96f)
        val width = unit * fontSize
        stream.saveGraphicsState()
        val state = PDExtendedGraphicsState()
        state.nonStrokingAlphaConstant = 0.16f
        stream.setGraphicsStateParameters(state)
        stream.setNonStrokingColor(110, 110, 110)
        stream.beginText()
        stream.setFont(font, fontSize)
        val cx = pw / 2; val cy = ph / 2
        val cos = kotlin.math.cos(angle); val sin = kotlin.math.sin(angle)
        // Start half the text's width back from the centre, along the diagonal.
        val tx = cx - cos * width / 2 + sin * fontSize * 0.3f
        val ty = cy - sin * width / 2 - cos * fontSize * 0.3f
        stream.setTextMatrix(Matrix(cos, sin, -sin, cos, tx, ty))
        stream.showText(text)
        stream.endText()
        stream.restoreGraphicsState()
    }

    /** "3 / 12" in small grey letters at the bottom centre. */
    private fun addPageNumber(stream: PDPageContentStream, label: String, pw: Float) {
        val font = PDType1Font.HELVETICA
        val fontSize = 9f
        val width = font.getStringWidth(label) / 1000f * fontSize
        stream.saveGraphicsState()
        stream.setNonStrokingColor(90, 90, 90)
        stream.beginText()
        stream.setFont(font, fontSize)
        stream.newLineAtOffset((pw - width) / 2, 5f)
        stream.showText(label)
        stream.endText()
        stream.restoreGraphicsState()
    }

    /** Rough size of the PDF for each quality, from a sample of up to 3 pages. */
    fun estimateBytes(pageFiles: List<File>, quality: PdfQuality): Long {
        if (pageFiles.isEmpty()) return 0
        val step = max(1, pageFiles.size / 3)
        val sample = pageFiles.filterIndexed { i, _ -> i % step == 0 }.take(3)
        val average = sample.sumOf { imageBytes(it, quality).size.toLong() } / sample.size
        return average * pageFiles.size + 1_500L * pageFiles.size
    }

    /** JPEG bytes for one page at the chosen quality. */
    private fun imageBytes(file: File, quality: PdfQuality): ByteArray {
        if (quality.maxSide == 0) return file.readBytes()

        val bounds = BitmapFactory.Options().apply { inJustDecodeBounds = true }
        BitmapFactory.decodeFile(file.path, bounds)
        val longest = max(bounds.outWidth, bounds.outHeight)
        if (longest <= 0) return file.readBytes()

        // Decode at a reduced size first (saves memory), then scale exactly.
        var sample = 1
        while (longest / (sample * 2) >= quality.maxSide) sample *= 2
        val decoded = BitmapFactory.decodeFile(file.path, BitmapFactory.Options().apply { inSampleSize = sample })
            ?: return file.readBytes()

        val scale = quality.maxSide.toFloat() / max(decoded.width, decoded.height)
        val bitmap = if (scale < 1f) {
            Bitmap.createScaledBitmap(
                decoded,
                (decoded.width * scale).roundToInt(),
                (decoded.height * scale).roundToInt(),
                true,
            ).also { if (it !== decoded) decoded.recycle() }
        } else decoded

        val stream = ByteArrayOutputStream()
        bitmap.compress(Bitmap.CompressFormat.JPEG, quality.jpegQuality, stream)
        bitmap.recycle()
        return stream.toByteArray()
    }

    /**
     * Turns each page of a PDF from the phone into an image, so it can be merged or edited
     * like a scan. Password-protected PDFs can't be opened.
     */
    fun renderPdfPages(context: Context, uri: Uri, onPage: (Bitmap) -> Unit): Int {
        val descriptor = context.contentResolver.openFileDescriptor(uri, "r")
            ?: error("Could not open the PDF")
        val renderer = try {
            PdfRenderer(descriptor)
        } catch (e: SecurityException) {
            descriptor.close()
            error("This PDF is password-protected")
        }
        try {
            for (i in 0 until renderer.pageCount) {
                val page = renderer.openPage(i)
                try {
                    // About 170 dpi, capped so huge pages don't run out of memory.
                    val scale = minOf(170f / 72f, 2200f / max(page.width, page.height))
                    val bitmap = Bitmap.createBitmap(
                        (page.width * scale).roundToInt().coerceAtLeast(1),
                        (page.height * scale).roundToInt().coerceAtLeast(1),
                        Bitmap.Config.ARGB_8888,
                    )
                    bitmap.eraseColor(Color.WHITE)
                    page.render(bitmap, null, null, PdfRenderer.Page.RENDER_MODE_FOR_DISPLAY)
                    onPage(bitmap)
                    bitmap.recycle()
                } finally {
                    page.close()
                }
            }
            return renderer.pageCount
        } finally {
            renderer.close()
            descriptor.close()
        }
    }

    /**
     * Copies a finished PDF into the phone's Downloads folder. Returns the file name used.
     * On Android 8-9 the caller must have asked for the storage permission first.
     */
    fun saveToDownloads(context: Context, pdf: File, displayName: String): String =
        saveToPublic(context, pdf, displayName, "application/pdf", image = false)

    /** Copies a page image into the phone's gallery (Pictures/My Scanner). Returns the file name used. */
    fun saveToGallery(context: Context, image: File, displayName: String): String =
        saveToPublic(context, image, displayName, "image/jpeg", image = true)

    private fun saveToPublic(context: Context, file: File, displayName: String, mime: String, image: Boolean): String {
        val extension = "." + displayName.substringAfterLast('.', "")
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.Q) {
            val resolver = context.contentResolver
            val values = ContentValues().apply {
                put(MediaStore.MediaColumns.DISPLAY_NAME, displayName)
                put(MediaStore.MediaColumns.MIME_TYPE, mime)
                put(
                    MediaStore.MediaColumns.RELATIVE_PATH,
                    if (image) Environment.DIRECTORY_PICTURES + "/My Scanner" else Environment.DIRECTORY_DOWNLOADS,
                )
                put(MediaStore.MediaColumns.IS_PENDING, 1)
            }
            val collection = if (image) MediaStore.Images.Media.EXTERNAL_CONTENT_URI else MediaStore.Downloads.EXTERNAL_CONTENT_URI
            val uri = resolver.insert(collection, values) ?: error("Could not create the file")
            val output = resolver.openOutputStream(uri) ?: error("Could not write the file")
            output.use { out -> file.inputStream().use { it.copyTo(out) } }
            values.clear()
            values.put(MediaStore.MediaColumns.IS_PENDING, 0)
            resolver.update(uri, values, null, null)
            return displayName
        } else {
            @Suppress("DEPRECATION")
            val base = Environment.getExternalStoragePublicDirectory(if (image) Environment.DIRECTORY_PICTURES else Environment.DIRECTORY_DOWNLOADS)
            val folder = if (image) File(base, "My Scanner") else base
            folder.mkdirs()
            var target = File(folder, displayName)
            var n = 1
            while (target.exists()) {
                target = File(folder, displayName.removeSuffix(extension) + " ($n)" + extension)
                n++
            }
            file.copyTo(target)
            if (image) {
                // Tell the gallery about the new picture.
                android.media.MediaScannerConnection.scanFile(context, arrayOf(target.path), arrayOf(mime), null)
            }
            return target.name
        }
    }

    /** Makes a title safe to use as a file name. */
    fun fileNameFor(title: String): String {
        val clean = title.replace(Regex("[\\\\/:*?\"<>|]"), " ").trim().ifEmpty { "Scan" }
        return "${clean.take(80)}.pdf"
    }

    /** A safe file name (no extension) from a title. */
    fun baseNameFor(title: String): String = fileNameFor(title).removeSuffix(".pdf")

    fun formatSize(bytes: Long): String = when {
        bytes >= 1_000_000 -> String.format("%.1f MB", bytes / 1_000_000.0)
        bytes >= 1_000 -> "${bytes / 1_000} KB"
        else -> "$bytes B"
    }
}
