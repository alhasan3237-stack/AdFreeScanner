package com.myscanner.app.scan

import kotlinx.coroutines.sync.Mutex

/**
 * Only one big image job (full-size page, zoom image, rotation) at a time.
 * Running several at once on a 3-4 GB phone is what makes the system kill the app.
 */
object HeavyWork {
    val lock = Mutex()
}
