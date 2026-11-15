package com.myscanner.app.scan

import android.content.Context
import android.util.Log
import java.io.File
import java.io.RandomAccessFile
import java.nio.MappedByteBuffer
import java.nio.channels.FileChannel

/**
 * Protects the app from crashes inside the image library's native code (these close the app
 * without any error report, so they can't be caught normally).
 *
 * Each risky part marks a byte in a small memory-mapped file while it runs. If the app dies in
 * the middle, the mark is still there at the next start, so that part is switched off for good and
 * the app keeps working without it (the photo is still cropped by the other methods).
 */
object NativeGuard {
    const val CORNER_NET = 1
    const val PAGE_NET = 2
    const val LIVE = 3
    private const val SLOTS = 8

    private var buf: MappedByteBuffer? = null
    private var prefs: android.content.SharedPreferences? = null
    private val counts = IntArray(SLOTS)

    /** What was switched off this time, for a short notice (null = nothing). */
    @Volatile var justDisabled: String? = null

    fun init(context: Context) {
        val p = context.getSharedPreferences("native_guard", Context.MODE_PRIVATE)
        prefs = p
        runCatching {
            val f = File(context.filesDir, "native_guard.bin")
            val raf = RandomAccessFile(f, "rw")
            raf.setLength(SLOTS.toLong())
            val b = raf.channel.map(FileChannel.MapMode.READ_WRITE, 0, SLOTS.toLong())
            // A mark left over = the app died inside that part last time. Blame the innermost one.
            val left = (1 until SLOTS).filter { b.get(it).toInt() != 0 }
            val blame = when {
                CORNER_NET in left -> CORNER_NET
                PAGE_NET in left -> PAGE_NET
                LIVE in left -> LIVE
                else -> null
            }
            if (blame != null) {
                p.edit().putBoolean("off_$blame", true).commit()
                justDisabled = name(blame)
                Log.e("NativeGuard", "Crashed inside ${name(blame)} last time; it is now off")
            }
            for (i in 0 until SLOTS) b.put(i, 0)
            buf = b
        }
        apply()
    }

    private fun name(slot: Int) = when (slot) {
        CORNER_NET -> "corner network"
        PAGE_NET -> "page network"
        else -> "live outline"
    }

    fun isOff(slot: Int) = prefs?.getBoolean("off_$slot", false) ?: false

    /** Switches every part back on (the user asked to try again). */
    fun resetAll() {
        prefs?.edit()?.clear()?.commit()
        apply()
    }

    private fun apply() {
        DocQuad.enabled = !isOff(CORNER_NET)
        PageNet.enabled = !isOff(PAGE_NET)
    }

    /** Runs [block] with [slot] marked as running. */
    inline fun <T> guard(slot: Int, block: () -> T): T {
        enter(slot)
        try {
            return block()
        } finally {
            leave(slot)
        }
    }

    @Synchronized fun enter(slot: Int) {
        counts[slot]++
        buf?.put(slot, 1)
    }

    @Synchronized fun leave(slot: Int) {
        counts[slot]--
        if (counts[slot] <= 0) { counts[slot] = 0; buf?.put(slot, 0) }
    }
}
