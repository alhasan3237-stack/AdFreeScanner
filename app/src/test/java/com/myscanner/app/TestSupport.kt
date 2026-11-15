package com.myscanner.app

import org.junit.Assume
import org.opencv.core.Core
import java.io.File

/** Helpers for the PC tests. */
object TestSupport {
    /** Folder with real test photos (set SCANNER_TESTDATA); tests that need it are skipped without it. */
    val dataDir: File? = System.getProperty("scanner.testdata")?.takeIf { it.isNotEmpty() }?.let { File(it) }?.takeIf { it.isDirectory }

    private var loaded: Boolean? = null

    /** Loads the desktop OpenCV library (set OPENCV_JVM_LIB to its folder). Skips the test if missing. */
    fun requireOpenCv() {
        if (loaded == null) {
            loaded = try { System.loadLibrary("opencv_java4"); Core.getVersionString(); true } catch (e: Throwable) { false }
        }
        Assume.assumeTrue("Desktop OpenCV not available (set OPENCV_JVM_LIB)", loaded == true)
    }

    fun requireData(): File {
        Assume.assumeTrue("Test photos not available (set SCANNER_TESTDATA)", dataDir != null)
        return dataDir!!
    }
}
