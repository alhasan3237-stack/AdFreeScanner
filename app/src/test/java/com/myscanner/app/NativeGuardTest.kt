package com.myscanner.app

import androidx.test.core.app.ApplicationProvider
import com.myscanner.app.scan.DocQuad
import com.myscanner.app.scan.NativeGuard
import org.junit.Assert.assertEquals
import org.junit.Assert.assertFalse
import org.junit.Assert.assertTrue
import org.junit.Test
import org.junit.runner.RunWith
import org.robolectric.RobolectricTestRunner

@RunWith(RobolectricTestRunner::class)
class NativeGuardTest {
    @Test
    fun aPartThatDiesIsSwitchedOff() {
        val app = ApplicationProvider.getApplicationContext<android.content.Context>()
        NativeGuard.init(app)
        NativeGuard.resetAll()
        NativeGuard.guard(NativeGuard.CORNER_NET) { } // finished normally: nothing is blamed
        NativeGuard.init(app)
        assertFalse(NativeGuard.isOff(NativeGuard.CORNER_NET)); assertTrue(DocQuad.enabled)
        // "Dies" inside the live outline while the corner network runs: the corner network is blamed.
        NativeGuard.enter(NativeGuard.LIVE); NativeGuard.enter(NativeGuard.CORNER_NET)
        NativeGuard.init(app)
        assertTrue(NativeGuard.isOff(NativeGuard.CORNER_NET)); assertFalse(DocQuad.enabled)
        assertFalse(NativeGuard.isOff(NativeGuard.LIVE))
        assertEquals("corner network", NativeGuard.justDisabled)
        NativeGuard.resetAll(); assertTrue(DocQuad.enabled)
    }
}
