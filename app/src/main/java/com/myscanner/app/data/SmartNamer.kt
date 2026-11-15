package com.myscanner.app.data

import com.google.mlkit.vision.text.Text
import java.text.SimpleDateFormat
import java.util.Calendar
import java.util.Date
import java.util.Locale

/**
 * Suggests a file name from the text on the first page, for example
 * "Electricity bill – Oct 2026", "Invoice – Star Tech – Mar 2026" or "Admit card – HSC 2026".
 *
 * It looks for: the kind of document (keywords), the biggest heading near the top, and a date.
 * Everything runs on the phone.
 */
object SmartNamer {

    private data class Kind(val label: String, val generic: Boolean, val keywords: List<String>)

    // Most specific first. "generic" kinds get the heading added (e.g. the shop's name).
    private val kinds = listOf(
        Kind("Electricity bill", false, listOf("electricity", "kwh", "desco", "dpdc", "bpdb", "nesco", "palli bidyut", "prepaid meter")),
        Kind("Gas bill", false, listOf("titas gas", "gas bill", "gas transmission")),
        Kind("Water bill", false, listOf("wasa", "water bill", "water supply")),
        Kind("Internet bill", false, listOf("internet bill", "broadband", "isp bill")),
        Kind("Bank statement", true, listOf("statement of account", "account statement", "bank statement")),
        Kind("NID card", false, listOf("national id", "nid no", "smart card", "national identity")),
        Kind("Birth certificate", false, listOf("birth registration", "birth certificate")),
        Kind("Passport", false, listOf("passport no", "passport")),
        Kind("Admit card", true, listOf("admit card")),
        Kind("Marksheet", true, listOf("marksheet", "mark sheet", "transcript", "grade sheet", "gpa")),
        Kind("Prescription", true, listOf("prescription", "mbbs", "diagnosis", "rx")),
        Kind("Invoice", true, listOf("invoice", "tax invoice")),
        Kind("Receipt", true, listOf("money receipt", "receipt", "cash memo")),
        Kind("Question paper", true, listOf("full marks", "answer all questions", "answer any")),
        Kind("Certificate", true, listOf("certificate", "certify that", "certified that")),
        Kind("Agreement", true, listOf("agreement", "contract", "deed")),
        Kind("CV", false, listOf("curriculum vitae", "resume")),
        Kind("Application", true, listOf("application for", "subject: application")),
        Kind("Notice", true, listOf("notice")),
        Kind("Letter", true, listOf("dear sir", "dear madam", "yours sincerely", "yours faithfully")),
        Kind("Bill", true, listOf("bill no", "amount due", "total payable")),
    )

    private val months = listOf("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec")
    private val monthRe = "(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\\.?"

    fun suggest(text: Text?, scannedAt: Long = System.currentTimeMillis()): String {
        val fallback = "Scan – " + SimpleDateFormat("d MMM yyyy, HH:mm", Locale.ENGLISH).format(Date(scannedAt))
        if (text == null || text.text.isBlank()) return fallback

        val lower = text.text.lowercase(Locale.ENGLISH)
        val kind = kinds.firstOrNull { k -> k.keywords.any { containsWord(lower, it) } }
        val heading = findHeading(text)
        val date = findDate(lower)

        val base = when {
            kind == null -> heading
            heading == null -> kind.label
            heading.contains(kind.label, ignoreCase = true) -> heading
            kind.generic && heading.length <= 28 &&
                !kind.keywords.any { heading.contains(it, ignoreCase = true) } -> "${kind.label} – $heading"
            else -> kind.label
        } ?: return fallback

        val datePart = date ?: SimpleDateFormat("d MMM yyyy", Locale.ENGLISH).format(Date(scannedAt))
        return "$base – $datePart".take(80)
    }

    private fun containsWord(text: String, word: String): Boolean =
        Regex("(^|[^a-z0-9])" + Regex.escape(word) + "($|[^a-z0-9])").containsMatchIn(text)

    /** The biggest line of real words in the top half of the page. */
    private fun findHeading(text: Text): String? {
        val lines = text.textBlocks.flatMap { it.lines }.filter { it.boundingBox != null }
        if (lines.isEmpty()) return null
        val pageBottom = lines.maxOf { it.boundingBox!!.bottom }.coerceAtLeast(1)

        return lines
            .asSequence()
            .filter { it.boundingBox!!.top < pageBottom * 0.5 }
            .map { it to clean(it.text) }
            .filter { (_, t) ->
                val letters = t.count { it.isLetter() }
                val words = t.split(' ').count { it.isNotBlank() }
                t.length in 4..48 && words in 1..7 && letters >= t.length * 0.7
            }
            .maxByOrNull { (line, t) ->
                val height = line.boundingBox!!.height().toDouble()
                val words = t.split(' ').count { it.isNotBlank() }.coerceAtMost(4)
                height * (1 + 0.15 * words)
            }
            ?.second
            ?.let { titleCaseIfShouting(it) }
    }

    private fun clean(s: String): String =
        s.replace(Regex("[|_*#=~^<>\\[\\]{}]"), " ")
            .replace(Regex("\\s+"), " ")
            .trim()
            .trim('.', ',', ':', ';', '-', '–')
            .trim()

    private fun titleCaseIfShouting(s: String): String {
        val letters = s.filter { it.isLetter() }
        if (letters.isEmpty() || letters.any { it.isLowerCase() }) return s
        return s.lowercase(Locale.ENGLISH).split(' ').joinToString(" ") { word ->
            if (word.length <= 3 && word in setOf("of", "and", "the", "for", "to", "in", "on", "at")) word
            else word.replaceFirstChar { it.titlecase(Locale.ENGLISH) }
        }.replaceFirstChar { it.titlecase(Locale.ENGLISH) }
    }

    /** First believable date in the text, as "Oct 2026". Day-first like Bangladesh (dd/mm/yyyy). */
    private fun findDate(lower: String): String? {
        val thisYear = Calendar.getInstance().get(Calendar.YEAR)
        fun ok(y: Int) = y in 1990..(thisYear + 1)
        fun label(m: Int, y: Int) = months[m - 1].replaceFirstChar { it.uppercase() } + " " + y
        fun year(s: String): Int = s.toInt().let { if (s.length == 2) 2000 + it else it }

        val patterns: List<Pair<Regex, (MatchResult) -> String?>> = listOf(
            // 2026-10-12
            Regex("\\b(\\d{4})[-/.](\\d{1,2})[-/.](\\d{1,2})\\b") to { m: MatchResult ->
                val y = m.groupValues[1].toInt(); val mo = m.groupValues[2].toInt()
                if (ok(y) && mo in 1..12) label(mo, y) else null
            },
            // 12/10/2026, 12-10-26
            Regex("\\b(\\d{1,2})[-/.](\\d{1,2})[-/.](\\d{2,4})\\b") to { m: MatchResult ->
                val d = m.groupValues[1].toInt(); val mo = m.groupValues[2].toInt(); val y = year(m.groupValues[3])
                if (ok(y) && mo in 1..12 && d in 1..31) label(mo, y) else null
            },
            // 12 Oct 2026, 12th October, 2026
            Regex("\\b(\\d{1,2})(?:st|nd|rd|th)?[\\s-]+$monthRe[\\s,.-]+(\\d{4})\\b") to { m: MatchResult ->
                val mo = months.indexOf(m.groupValues[2]) + 1; val y = m.groupValues[3].toInt()
                if (ok(y) && mo > 0) label(mo, y) else null
            },
            // October 12, 2026
            Regex("\\b$monthRe\\s+(\\d{1,2})(?:st|nd|rd|th)?,?\\s+(\\d{4})\\b") to { m: MatchResult ->
                val mo = months.indexOf(m.groupValues[1]) + 1; val y = m.groupValues[3].toInt()
                if (ok(y) && mo > 0) label(mo, y) else null
            },
            // October 2026, Oct-2026
            Regex("\\b$monthRe[\\s,'-]+(\\d{4})\\b") to { m: MatchResult ->
                val mo = months.indexOf(m.groupValues[1]) + 1; val y = m.groupValues[2].toInt()
                if (ok(y) && mo > 0) label(mo, y) else null
            },
        )
        for ((re, toLabel) in patterns) {
            for (m in re.findAll(lower)) toLabel(m)?.let { return it }
        }
        return null
    }
}
