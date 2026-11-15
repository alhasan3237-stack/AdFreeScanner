package com.myscanner.app.ui.components

import android.content.Context
import android.content.Intent
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.navigationBarsPadding
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.selection.selectable
import androidx.compose.foundation.selection.selectableGroup
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Button
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalBottomSheet
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.material3.rememberModalBottomSheetState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.semantics.Role
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.input.ImeAction
import androidx.compose.ui.text.input.KeyboardCapitalization
import androidx.compose.ui.unit.dp
import androidx.core.content.FileProvider
import com.myscanner.app.data.PdfOptions
import com.myscanner.app.data.PdfQuality
import androidx.compose.material3.Switch
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.text.input.PasswordVisualTransformation
import com.myscanner.app.data.PdfTools
import com.myscanner.app.data.PdfPageSize
import androidx.compose.foundation.layout.imePadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.FilterChip
import java.io.File

/** What the Save sheet will do once a size is picked. */
enum class SheetAction(val title: String, val button: String) {
    SAVE("Save PDF", "Save PDF"),
    SHARE("Share PDF", "Share PDF"),
}

/**
 * The size picker shown when tapping Save PDF or Share:
 * Original / Medium / Small, each with its estimated file size.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun SavePdfSheet(
    action: SheetAction,
    defaultQuality: PdfQuality,
    estimates: Map<PdfQuality, Long>?,
    onConfirm: (PdfOptions) -> Unit,
    onDismiss: () -> Unit,
) {
    var selected by remember { mutableStateOf(defaultQuality) }
    var searchable by remember { mutableStateOf(true) }
    var protect by remember { mutableStateOf(false) }
    var password by remember { mutableStateOf("") }
    var more by remember { mutableStateOf(false) }
    var pageSize by remember { mutableStateOf(PdfPageSize.FIT) }
    var pageNumbers by remember { mutableStateOf(false) }
    var watermark by remember { mutableStateOf("") }
    ModalBottomSheet(
        onDismissRequest = onDismiss,
        sheetState = rememberModalBottomSheetState(skipPartiallyExpanded = true),
    ) {
        Column(
            Modifier
                .fillMaxWidth()
                // Scrolls, and moves up with the keyboard: the password box and the button stay reachable.
                .verticalScroll(rememberScrollState())
                .padding(horizontal = 24.dp)
                .navigationBarsPadding()
                .imePadding()
                .padding(bottom = 16.dp)
        ) {
            Text(action.title, style = MaterialTheme.typography.titleLarge, fontWeight = FontWeight.SemiBold)
            Text(
                "Choose a file size",
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant,
            )
            Spacer(Modifier.height(12.dp))
            Column(Modifier.selectableGroup()) {
                PdfQuality.entries.forEach { quality ->
                    Row(
                        Modifier
                            .fillMaxWidth()
                            .height(64.dp)
                            .selectable(
                                selected = selected == quality,
                                onClick = { selected = quality },
                                role = Role.RadioButton,
                            ),
                        verticalAlignment = Alignment.CenterVertically,
                    ) {
                        RadioButton(selected = selected == quality, onClick = null)
                        Column(Modifier.weight(1f).padding(start = 12.dp)) {
                            Text(quality.label, style = MaterialTheme.typography.bodyLarge, fontWeight = FontWeight.Medium)
                            Text(
                                quality.description,
                                style = MaterialTheme.typography.bodySmall,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                        val size = estimates?.get(quality)
                        if (size == null) {
                            CircularProgressIndicator(Modifier.padding(4.dp).size(18.dp), strokeWidth = 2.dp)
                        } else {
                            Text(
                                "≈ " + PdfTools.formatSize(size),
                                style = MaterialTheme.typography.bodyMedium,
                                color = MaterialTheme.colorScheme.onSurfaceVariant,
                            )
                        }
                    }
                }
            }
            Spacer(Modifier.height(8.dp))
            OptionSwitch("Searchable text", "Find and copy words in any PDF app", searchable) { searchable = it }
            OptionSwitch("Protect with password", "Needs the password to open", protect) { protect = it }
            if (protect) {
                OutlinedTextField(
                    value = password,
                    onValueChange = { password = it },
                    singleLine = true,
                    label = { Text("Password") },
                    visualTransformation = PasswordVisualTransformation(),
                    keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Password),
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            if (!more) {
                TextButton(onClick = { more = true }) { Text("More options: paper size, page numbers, watermark") }
            } else {
                Spacer(Modifier.height(8.dp))
                Text("Paper size", style = MaterialTheme.typography.bodyLarge)
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    PdfPageSize.entries.forEach { size ->
                        FilterChip(selected = pageSize == size, onClick = { pageSize = size }, label = { Text(size.label) })
                    }
                }
                OptionSwitch("Page numbers", "Small “2 / 5” at the bottom of each page", pageNumbers) { pageNumbers = it }
                OutlinedTextField(
                    value = watermark,
                    onValueChange = { watermark = it.take(40) },
                    singleLine = true,
                    label = { Text("Watermark (optional)") },
                    placeholder = { Text("e.g. COPY") },
                    modifier = Modifier.fillMaxWidth(),
                )
            }
            Spacer(Modifier.height(16.dp))
            Button(
                onClick = {
                    onConfirm(
                        PdfOptions(
                            quality = selected, searchable = searchable, password = if (protect) password else null,
                            pageSize = pageSize, pageNumbers = pageNumbers, watermark = watermark.trim().ifEmpty { null },
                        )
                    )
                },
                enabled = !protect || password.isNotEmpty(),
                modifier = Modifier.fillMaxWidth().height(52.dp),
            ) { Text(action.button, style = MaterialTheme.typography.titleMedium) }
        }
    }
}

@Composable
private fun OptionSwitch(title: String, subtitle: String, checked: Boolean, onChange: (Boolean) -> Unit) {
    Row(
        Modifier.fillMaxWidth().height(56.dp).selectable(selected = checked, onClick = { onChange(!checked) }, role = Role.Switch),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyLarge)
            Text(subtitle, style = MaterialTheme.typography.bodySmall, color = MaterialTheme.colorScheme.onSurfaceVariant)
        }
        Switch(checked = checked, onCheckedChange = null)
    }
}

@Composable
fun TextInputDialog(
    title: String,
    initial: String,
    confirmLabel: String,
    onConfirm: (String) -> Unit,
    onDismiss: () -> Unit,
) {
    var text by remember { mutableStateOf(initial) }
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = {
            OutlinedTextField(
                value = text,
                onValueChange = { text = it },
                singleLine = true,
                label = { Text("Name") },
                keyboardOptions = KeyboardOptions(
                    capitalization = KeyboardCapitalization.Sentences,
                    imeAction = ImeAction.Done,
                ),
            )
        },
        confirmButton = { TextButton(onClick = { onConfirm(text) }) { Text(confirmLabel) } },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

@Composable
fun ConfirmDialog(
    title: String,
    message: String,
    confirmLabel: String,
    onConfirm: () -> Unit,
    onDismiss: () -> Unit,
) {
    AlertDialog(
        onDismissRequest = onDismiss,
        title = { Text(title) },
        text = { Text(message) },
        confirmButton = {
            TextButton(onClick = onConfirm) {
                Text(confirmLabel, color = MaterialTheme.colorScheme.error)
            }
        },
        dismissButton = { TextButton(onClick = onDismiss) { Text("Cancel") } },
    )
}

/** A full-screen-ish busy message, e.g. "Building PDF…". */
@Composable
fun BusyDialog(message: String) {
    AlertDialog(
        onDismissRequest = {},
        confirmButton = {},
        text = {
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(16.dp)) {
                CircularProgressIndicator()
                Text(message)
            }
        },
    )
}

/** Opens Android's share sheet for a PDF file. */
fun sharePdf(context: Context, file: File) {
    val uri = FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", file)
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "application/pdf"
        putExtra(Intent.EXTRA_STREAM, uri)
        addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    }
    context.startActivity(Intent.createChooser(intent, "Share PDF"))
}

fun shareText(context: Context, text: String) {
    val intent = Intent(Intent.ACTION_SEND).apply {
        type = "text/plain"
        putExtra(Intent.EXTRA_TEXT, text)
    }
    context.startActivity(Intent.createChooser(intent, "Share text"))
}

/** Opens Android's share sheet for one or more page pictures. */
fun shareImages(context: Context, files: List<File>) {
    if (files.isEmpty()) return
    val uris = ArrayList(files.map { FileProvider.getUriForFile(context, "${context.packageName}.fileprovider", it) })
    val intent = if (uris.size == 1) {
        Intent(Intent.ACTION_SEND).apply { putExtra(Intent.EXTRA_STREAM, uris[0]) }
    } else {
        Intent(Intent.ACTION_SEND_MULTIPLE).apply { putParcelableArrayListExtra(Intent.EXTRA_STREAM, uris) }
    }
    intent.type = "image/jpeg"
    intent.addFlags(Intent.FLAG_GRANT_READ_URI_PERMISSION)
    context.startActivity(Intent.createChooser(intent, "Share pictures"))
}

/** Sends a PDF to Android's print screen (a printer, or "Save as PDF"). Returns false if printing isn't available. */
fun printPdf(context: Context, file: File, title: String): Boolean {
    val manager = context.getSystemService(Context.PRINT_SERVICE) as? android.print.PrintManager ?: return false
    val adapter = object : android.print.PrintDocumentAdapter() {
        override fun onLayout(
            oldAttributes: android.print.PrintAttributes?, newAttributes: android.print.PrintAttributes,
            cancellationSignal: android.os.CancellationSignal?, callback: LayoutResultCallback, extras: android.os.Bundle?,
        ) {
            if (cancellationSignal?.isCanceled == true) { callback.onLayoutCancelled(); return }
            val info = android.print.PrintDocumentInfo.Builder(file.name)
                .setContentType(android.print.PrintDocumentInfo.CONTENT_TYPE_DOCUMENT)
                .setPageCount(android.print.PrintDocumentInfo.PAGE_COUNT_UNKNOWN)
                .build()
            callback.onLayoutFinished(info, true)
        }

        override fun onWrite(
            pages: Array<out android.print.PageRange>, destination: android.os.ParcelFileDescriptor,
            cancellationSignal: android.os.CancellationSignal?, callback: WriteResultCallback,
        ) {
            try {
                file.inputStream().use { input ->
                    java.io.FileOutputStream(destination.fileDescriptor).use { output -> input.copyTo(output) }
                }
                callback.onWriteFinished(arrayOf(android.print.PageRange.ALL_PAGES))
            } catch (e: Exception) {
                callback.onWriteFailed(e.message)
            }
        }
    }
    return runCatching { manager.print(title, adapter, null) }.isSuccess
}
