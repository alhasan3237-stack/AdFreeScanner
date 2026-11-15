package com.myscanner.app.ui.settings

import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.automirrored.outlined.ArrowBack
import androidx.compose.material.icons.outlined.VerifiedUser
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Card
import androidx.compose.material3.CardDefaults
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.HorizontalDivider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Switch
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.dp
import androidx.lifecycle.compose.collectAsStateWithLifecycle
import com.myscanner.app.ScannerApp
import com.myscanner.app.data.AppSettings
import com.myscanner.app.data.PdfQuality
import com.myscanner.app.data.PdfTools
import androidx.compose.runtime.LaunchedEffect
import com.myscanner.app.data.PhotoQuality
import com.myscanner.app.data.ScanFilter
import com.myscanner.app.data.ThemeMode
import kotlinx.coroutines.launch

private enum class Picker { PHOTO, FILTER, QUALITY, THEME }

@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SettingsScreen(app: ScannerApp, onBack: () -> Unit) {
    val store = app.settings
    val settings by store.settings.collectAsStateWithLifecycle(initialValue = AppSettings())
    val scope = rememberCoroutineScope()
    var picker by remember { mutableStateOf<Picker?>(null) }
    var showPrivacy by remember { mutableStateOf(false) }
    /** (all scans, kept original photos) in bytes; null while being measured. */
    var storage by remember { mutableStateOf<Pair<Long, Long>?>(null) }
    var confirmDeleteOriginals by remember { mutableStateOf(false) }
    LaunchedEffect(Unit) { storage = runCatching { app.repository.storageUse() }.getOrNull() }
    val version = remember {
        runCatching { app.packageManager.getPackageInfo(app.packageName, 0).versionName }.getOrNull() ?: "1.0"
    }

    Scaffold(
        topBar = {
            TopAppBar(
                title = { Text("Settings") },
                navigationIcon = {
                    IconButton(onClick = onBack) { Icon(Icons.AutoMirrored.Outlined.ArrowBack, "Back") }
                },
            )
        },
    ) { padding ->
        Column(
            Modifier
                .padding(padding)
                .fillMaxSize()
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 16.dp, vertical = 8.dp),
            verticalArrangement = Arrangement.spacedBy(18.dp),
        ) {
            Group("Scanning") {
                ValueRow("Photo quality", settings.photoQuality.description, settings.photoQuality.label) {
                    picker = Picker.PHOTO
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                SwitchRow("Auto-enhance animation", "Show the page being cleaned after each photo", settings.enhanceAnimation) {
                    scope.launch { store.setEnhanceAnimation(it) }
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                SwitchRow(
                    "Live edge detection",
                    "Show the page outline on the camera while you aim. Off: saves battery, corners are found after each photo",
                    settings.liveOutline,
                ) {
                    scope.launch { store.setLiveOutline(it) }
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                SwitchRow(
                    "Auto-capture",
                    if (settings.liveOutline) "Take the photo by itself when the page is steady"
                    else "Needs live edge detection",
                    settings.autoCapture && settings.liveOutline,
                ) {
                    if (settings.liveOutline) scope.launch { store.setAutoCapture(it) }
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                ValueRow("Default filter", settings.defaultFilter.description, settings.defaultFilter.label) {
                    picker = Picker.FILTER
                }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                Row(
                    Modifier
                        .fillMaxWidth()
                        .heightIn(min = 60.dp)
                        .clickable { scope.launch { store.setSmartNames(!settings.smartNames) } }
                        .padding(horizontal = 16.dp, vertical = 8.dp),
                    verticalAlignment = Alignment.CenterVertically,
                ) {
                    Column(Modifier.weight(1f)) {
                        Text("Smart file names", style = MaterialTheme.typography.bodyLarge)
                        Text(
                            "Name scans from their text, e.g. “Electricity bill – Oct 2026”",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                    Switch(
                        checked = settings.smartNames,
                        onCheckedChange = { on -> scope.launch { store.setSmartNames(on) } },
                    )
                }
            }

            Group("Saving PDFs") {
                ValueRow("Default size", settings.defaultQuality.description, settings.defaultQuality.label) {
                    picker = Picker.QUALITY
                }
            }

            Group("Storage") {
                SwitchRow(
                    "Keep original photos",
                    "Lets you re-crop and change the filter of saved pages later. Turn off to use about half the space.",
                    settings.keepOriginals,
                ) { scope.launch { store.setKeepOriginals(it) } }
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                ValueRow(
                    "Space used by scans",
                    storage?.let { "Of this, kept original photos: ${PdfTools.formatSize(it.second)}" } ?: "Measuring…",
                    storage?.let { PdfTools.formatSize(it.first) } ?: "",
                ) {}
                if ((storage?.second ?: 0L) > 0L) {
                    HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                    ValueRow("Delete kept original photos", "Frees space. Saved pages stay, but can't be re-cropped.", "") {
                        confirmDeleteOriginals = true
                    }
                }
            }

            Group("Text recognition") {
                ValueRow("Language", "Works offline", "English") {}
                HorizontalDivider(color = MaterialTheme.colorScheme.outlineVariant)
                ValueRow("Bangla text", "Optional download", "Coming later") {}
            }

            Group("Appearance") {
                ValueRow("Theme", null, settings.themeMode.label) { picker = Picker.THEME }
            }

            Group("About") {
                Column(Modifier.padding(16.dp)) {
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        Icon(Icons.Outlined.VerifiedUser, null, tint = MaterialTheme.colorScheme.primary)
                        Spacer(Modifier.width(10.dp))
                        Text("No ads. No tracking. No account.", fontWeight = FontWeight.SemiBold)
                    }
                    Text(
                        "Your scans and text never leave this phone unless you share them.",
                        style = MaterialTheme.typography.bodyMedium,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                    Text(
                        "Page-corner network based on DocQuadNet-256 from the MakeACopy app " +
                            "(github.com/egdels/makeacopy, Apache License 2.0), retrained for this app. " +
                            "Also uses OpenCV (Apache 2.0), PdfBox-Android (Apache 2.0) and Google ML Kit.",
                        style = MaterialTheme.typography.bodySmall,
                        color = MaterialTheme.colorScheme.onSurfaceVariant,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                    Row(
                        Modifier.fillMaxWidth().padding(top = 4.dp),
                        horizontalArrangement = Arrangement.SpaceBetween,
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        TextButton(onClick = { showPrivacy = true }) { Text("Privacy policy") }
                        Text(
                            "Version $version",
                            style = MaterialTheme.typography.bodySmall,
                            color = MaterialTheme.colorScheme.onSurfaceVariant,
                        )
                    }
                }
            }
            Spacer(Modifier.padding(8.dp))
        }
    }

    when (picker) {
        Picker.PHOTO -> ChoiceDialog(
            title = "Photo quality",
            options = PhotoQuality.entries,
            selected = settings.photoQuality,
            label = { "${it.label} — ${it.description}" },
            onPick = { scope.launch { store.setPhotoQuality(it) }; picker = null },
            onDismiss = { picker = null },
        )
        Picker.FILTER -> ChoiceDialog(
            title = "Default filter",
            options = ScanFilter.entries,
            selected = settings.defaultFilter,
            label = { "${it.label} — ${it.description}" },
            onPick = { scope.launch { store.setDefaultFilter(it) }; picker = null },
            onDismiss = { picker = null },
        )
        Picker.QUALITY -> ChoiceDialog(
            title = "Default PDF size",
            options = PdfQuality.entries,
            selected = settings.defaultQuality,
            label = { "${it.label} — ${it.description}" },
            onPick = { scope.launch { store.setDefaultQuality(it) }; picker = null },
            onDismiss = { picker = null },
        )
        Picker.THEME -> ChoiceDialog(
            title = "Theme",
            options = ThemeMode.entries,
            selected = settings.themeMode,
            label = { it.label },
            onPick = { scope.launch { store.setThemeMode(it) }; picker = null },
            onDismiss = { picker = null },
        )
        null -> Unit
    }

    if (confirmDeleteOriginals) {
        com.myscanner.app.ui.components.ConfirmDialog(
            title = "Delete the kept original photos?",
            message = "Your saved pages and PDFs stay exactly as they are. You only lose the ability to re-crop or change the filter of pages saved until now.",
            confirmLabel = "Delete",
            onConfirm = {
                confirmDeleteOriginals = false
                scope.launch {
                    runCatching { app.repository.deleteOriginals() }
                    storage = runCatching { app.repository.storageUse() }.getOrNull()
                }
            },
            onDismiss = { confirmDeleteOriginals = false },
        )
    }

    if (showPrivacy) {
        AlertDialog(
            onDismissRequest = { showPrivacy = false },
            title = { Text("Privacy policy") },
            text = {
                Text(
                    "This app shows no ads and collects no personal data. It has no account and no analytics. " +
                        "Scanning, cleaning, PDF creation and text recognition all happen on your phone. Your documents " +
                        "are stored only in the app's private storage and leave the phone only when you choose " +
                        "to save or share them."
                )
            },
            confirmButton = { TextButton(onClick = { showPrivacy = false }) { Text("OK") } },
        )
    }
}

@Composable
private fun Group(title: String, content: @Composable () -> Unit) {
    Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
        Text(
            title,
            style = MaterialTheme.typography.labelLarge,
            color = MaterialTheme.colorScheme.onSurfaceVariant,
            modifier = Modifier.padding(start = 4.dp),
        )
        Card(
            shape = RoundedCornerShape(16.dp),
            colors = CardDefaults.cardColors(containerColor = MaterialTheme.colorScheme.surfaceContainerLowest),
        ) {
            Column { content() }
        }
    }
}

@Composable
private fun SwitchRow(title: String, subtitle: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .heightIn(min = 60.dp)
            .clickable { onChange(!checked) }
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = onChange)
    }
}

@Composable
private fun ValueRow(title: String, subtitle: String?, value: String, onClick: () -> Unit) {
    Row(
        Modifier
            .fillMaxWidth()
            .heightIn(min = 60.dp)
            .clickable(onClick = onClick)
            .padding(horizontal = 16.dp, vertical = 8.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            if (subtitle != null) {
                Text(
                    subtitle,
                    style = MaterialTheme.typography.bodySmall,
                    color = MaterialTheme.colorScheme.onSurfaceVariant,
                )
            }
        }
        Text(value, style = MaterialTheme.typography.bodyMedium, color = MaterialTheme.colorScheme.onSurfaceVariant)
    }
}

@Composable
private fun <T> ChoiceDialog(
    title: String,
    options: List<T>,
    selected: T,
    label: (T) -> String,
    onPick: (T) -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            Column(Modifier.selectableGroup()) {
                options.forEach { option ->
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .heightIn(min = 52.dp)
                            .selectable(selected = option == selected, onClick = { onPick(option) }, role = Role.RadioButton),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        RadioButton(selected = option == selected, onClick = null)
                        Text(label(option), modifier = Modifier.padding(start = 12.dp))
                    }
                }
            }
        },
        confirmButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}
