package com.myscanner.app.ui.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Shapes
import androidx.compose.material3.darkColorScheme
import androidx.compose.material3.lightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import com.myscanner.app.data.ThemeMode

// Colors from the mockup: calm teal accent on a soft grey ground.
private val Teal = Color(0xFF0B6E68)
private val TealLight = Color(0xFF7FD4CC)

private val LightColors = lightColorScheme(
    primary = Teal,
    onPrimary = Color.White,
    primaryContainer = Color(0xFFCDEDEA),
    onPrimaryContainer = Color(0xFF00201E),
    secondary = Color(0xFF4A6361),
    background = Color(0xFFF5F7F8),
    onBackground = Color(0xFF16181D),
    surface = Color(0xFFF5F7F8),
    onSurface = Color(0xFF16181D),
    surfaceContainerLowest = Color.White,
    surfaceContainerLow = Color.White,
    surfaceContainer = Color(0xFFEEF1F2),
    onSurfaceVariant = Color(0xFF555B66),
    outline = Color(0xFF9AA1AB),
    outlineVariant = Color(0xFFDDE1E5),
    error = Color(0xFFB42318),
)

private val DarkColors = darkColorScheme(
    primary = TealLight,
    onPrimary = Color(0xFF003733),
    primaryContainer = Color(0xFF00504B),
    onPrimaryContainer = Color(0xFFA0F0E8),
    background = Color(0xFF111416),
    onBackground = Color(0xFFE2E3E5),
    surface = Color(0xFF111416),
    onSurface = Color(0xFFE2E3E5),
    surfaceContainerLowest = Color(0xFF1B1F22),
    surfaceContainerLow = Color(0xFF1B1F22),
    surfaceContainer = Color(0xFF22272A),
    onSurfaceVariant = Color(0xFFB4BAC2),
    outline = Color(0xFF6F7680),
    outlineVariant = Color(0xFF353B40),
    error = Color(0xFFFFB4AB),
)

private val AppShapes = Shapes(
    small = RoundedCornerShape(10.dp),
    medium = RoundedCornerShape(16.dp),
    large = RoundedCornerShape(20.dp),
)

@Composable
fun ScannerTheme(themeMode: ThemeMode, content: @Composable () -> Unit) {
    val dark = when (themeMode) {
        ThemeMode.SYSTEM -> isSystemInDarkTheme()
        ThemeMode.LIGHT -> false
        ThemeMode.DARK -> true
    }
    MaterialTheme(
        colorScheme = if (dark) DarkColors else LightColors,
        shapes = AppShapes,
        content = content,
    )
}
