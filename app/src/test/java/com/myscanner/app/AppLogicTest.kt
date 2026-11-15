package com.myscanner.app

import android.content.Context
import android.graphics.Bitmap
import android.graphics.Color
import androidx.test.core.app.ApplicationProvider
import com.myscanner.app.data.PdfOptions
import com.myscanner.app.data.PdfPageSize
import com.myscanner.app.data.PdfQuality
import com.myscanner.app.data.PdfTools
import com.myscanner.app.data.PhotoQuality
import com.myscanner.app.data.ScanFilter
import com.myscanner.app.scan.ImageProcessor
import com.myscanner.app.scan.Mark
import com.myscanner.app.scan.Marks
import com.myscanner.app.scan.P
import com.myscanner.app.scan.Quad
import com.myscanner.app.scan.ScanSession
import com.tom_roush.pdfbox.pdmodel.PDDocument
import org.junit.Assert.assertEquals
import org.junit.Assert.assertNotNull
import org.junit.Assert.assertNull
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.opencv.core.Core
import org.opencv.core.CvType
import org.opencv.core.Mat
import org.opencv.core.Scalar
import org.robolectric.RobolectricTestRunner
import java.io.File

/** The parts of the app that don't need a camera: unfinished scans, drawings, PDF options. */
@RunWith(RobolectricTestRunner::class)
class AppLogicTest {

    private val context: Context get() = ApplicationProvider.getApplicationContext()

    private val sampleMarks = listOf(
        Mark.Stroke(listOf(P(0.1f, 0.2f), P(0.5f, 0.25f), P(0.9f, 0.2f)), 0xFF1E4FD8.toInt(), 0.006f, false),
        Mark.Stroke(listOf(P(0.2f, 0.6f), P(0.8f, 0.6f)), 0xFFFFF176.toInt(), 0.035f, true),
        Mark.Label("Paid 12/11\nসই", 0.5f, 0.8f, 0.04f, 0xFFD32F2F.toInt()),
    )

    @Test
    fun marksSurviveSavingAndRotation() {
        val back = Marks.decode(Marks.encode(sampleMarks))
        assertEquals(3, back.size)
        assertEquals((sampleMarks[2] as Mark.Label).text, (back[2] as Mark.Label).text)
        assertEquals(sampleMarks[1], back[1])
        // Four quarter turns bring every mark back to where it was.
        var turned = back
        repeat(4) { turned = Marks.rotate(turned, 90) }
        val a = turned[0] as Mark.Stroke; val b = back[0] as Mark.Stroke
        a.points.zip(b.points).forEach { (p, q) -> assertEquals(q.x, p.x, 1e-4f); assertEquals(q.y, p.y, 1e-4f) }
        assertEquals(0, (turned[2] as Mark.Label).angle)
        // One turn clockwise: the top-left corner goes to the top-right.
        val once = Marks.rotate(listOf(Mark.Stroke(listOf(P(0f, 0f)), 0, 0.01f, false)), 90)[0] as Mark.Stroke
        assertEquals(1f, once.points[0].x, 1e-4f); assertEquals(0f, once.points[0].y, 1e-4f)
        assertTrue(Marks.decode("garbage|line\n\nS|oops").isEmpty())
    }

    @Test
    fun marksAreDrawnOnThePage() {
        TestSupport.requireOpenCv()
        val white = Mat(800, 600, CvType.CV_8UC3, Scalar(255.0, 255.0, 255.0))
        val out = ImageProcessor.withMarks(white, sampleMarks)
        // Pen: blue ink where the line passes. Highlighter: yellow tint. Text: some red pixels around its centre.
        val pen = out.get(180, 300) // on the pen line near its middle (y = 0.2..0.25 * 800)
        val hl = out.get(480, 300)
        assertTrue("highlighter tints yellow", hl[0] < 200 && hl[1] > 200 && hl[2] > 200)
        val penArea = out.submat(150, 210, 60, 540)
        val textArea = out.submat(560, 720, 100, 500)
        val minPen = Core.minMaxLoc(Mat().also { Core.extractChannel(penArea, it, 2) }).minVal
        val minText = Core.minMaxLoc(Mat().also { Core.extractChannel(textArea, it, 0) }).minVal
        println("MARKS pen centre=${pen?.toList()} min red in pen area=$minPen, min blue in text area=$minText")
        assertTrue("pen draws", minPen < 120)
        assertTrue("text draws", minText < 150)
        // A grey page becomes colour so the ink keeps its colour.
        val grey = Mat(400, 300, CvType.CV_8UC1, Scalar(255.0))
        assertEquals(3, ImageProcessor.withMarks(grey, sampleMarks).channels())
    }

    @Test
    fun unfinishedScanComesBackAfterTheAppIsClosed() {
        ScanSession.root(context).deleteRecursively()
        val s = ScanSession(context, targetDocumentId = 7L, defaultFilter = ScanFilter.MAGIC, photoQuality = PhotoQuality.STANDARD)
        val ids = (0 until 4).map { i ->
            val f = s.newPhotoFile().apply { writeBytes(byteArrayOf(1, 2, 3, i.toByte())) }
            s.add(f, Quad(P(0.1f, 0.1f), P(0.9f, 0.12f), P(0.88f, 0.9f), P(0.1f, 0.92f)), detected = i != 2)
        }
        s.update(ids[1]) { it.copy(filter = ScanFilter.BW, marks = sampleMarks) }
        s.rotate(ids[1])
        s.remove(ids[0])
        s.move(ids[3], -1) // order is now 1, 3, 2
        val before = s.pages.value

        val back = ScanSession.restore(context)
        assertNotNull(back); back!!
        assertEquals(7L, back.targetDocumentId)
        assertEquals(ScanFilter.MAGIC, back.defaultFilter)
        assertEquals(PhotoQuality.STANDARD, back.photoQuality)
        assertEquals(before.map { it.id }, back.pages.value.map { it.id })
        assertEquals(before.map { it.file.name }, back.pages.value.map { it.file.name })
        val p = back.pages.value[0]
        assertEquals(ScanFilter.BW, p.filter); assertEquals(90, p.rotation); assertEquals(3, p.marks.size)
        assertEquals(before[0].quad.key(), p.quad.key())
        assertEquals(false, back.pages.value[2].detected)
        // A new page gets a fresh id, never one that is in use.
        val newId = back.add(back.newPhotoFile().apply { writeBytes(byteArrayOf(9)) }, Quad.FULL, true)
        assertTrue(newId !in before.map { it.id })
        // A page whose photo is gone is left out; a discarded scan doesn't come back.
        back.pages.value[1].file.delete()
        assertEquals(3, ScanSession.restore(context)!!.pages.value.size)
        ScanSession.restore(context)!!.discard()
        assertNull(ScanSession.restore(context))
    }

    private fun pageJpeg(name: String, w: Int, h: Int): File {
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888).apply { eraseColor(Color.WHITE) }
        return File(context.cacheDir, name).also { f -> f.outputStream().use { bmp.compress(Bitmap.CompressFormat.JPEG, 90, it) } }
    }

    @Test
    fun pdfOptionsWork() {
        com.tom_roush.pdfbox.android.PDFBoxResourceLoader.init(context)
        val pages = listOf(pageJpeg("p1.jpg", 1200, 1700), pageJpeg("p2.jpg", 1700, 1200))
        val out = File(context.cacheDir, "t.pdf")

        PdfTools.buildPdf(pages, PdfOptions(PdfQuality.MEDIUM, searchable = false), out)
        PDDocument.load(out).use { doc ->
            assertEquals(2, doc.numberOfPages)
            val box = doc.getPage(0).mediaBox
            assertEquals(595f, box.width, 0.5f); assertEquals(595f * 1700 / 1200, box.height, 1f) // as tall as the scan
        }

        val full = PdfOptions(PdfQuality.SMALL, searchable = false, pageSize = PdfPageSize.A4, pageNumbers = true, watermark = "COPY – নকল")
        PdfTools.buildPdf(pages, full, out)
        PDDocument.load(out).use { doc ->
            val a = doc.getPage(0).mediaBox; val b = doc.getPage(1).mediaBox
            assertEquals(595f, a.width, 0.5f); assertEquals(842f, a.height, 0.5f)
            assertEquals(842f, b.width, 0.5f); assertEquals(595f, b.height, 0.5f) // a landscape scan gets landscape paper
            val text = com.tom_roush.pdfbox.text.PDFTextStripper().getText(doc)
            println("PDF text: " + text.replace("\n", " | "))
            assertTrue("page numbers", text.contains("1 / 2") && text.contains("2 / 2"))
            assertTrue("watermark", text.contains("COPY"))
        }

        PdfTools.buildPdf(pages, PdfOptions(PdfQuality.SMALL, searchable = false, password = "secret"), out)
        assertTrue("needs the password", runCatching { PDDocument.load(out).close() }.isFailure)
        PDDocument.load(out, "secret").use { assertEquals(2, it.numberOfPages) }
    }
}
