package com.myscanner.app

import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.runtime.getValue
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.myscanner.app.data.AppSettings
import com.myscanner.app.ui.AppNavigation
import com.myscanner.app.ui.theme.ScannerTheme

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()
        val app = application as ScannerApp
        setContent {
            val settings by app.settings.settings.collectAsStateWithLifecycle(initialValue = AppSettings())
            ScannerTheme(themeMode = settings.themeMode) {
                AppNavigation(app = app)
            }
        }
    }
}
