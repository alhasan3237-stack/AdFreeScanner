# My Scanner — ad-free document scanner for Android

## New in version 8

**Faster**
- Use the **release** build for daily use: in Android Studio choose *Build > Select Build Variant > release*, then Run. The debug build is 2-5x slower.
- The live page finder rests between looks and stops while a photo is taken, so the camera stays smooth.
- Shadow removal is about 6x faster with exactly the same result; pages save about 3x faster.
- Lists use small thumbnail files; previews are sized for your screen and neighbouring pages are prepared ahead.
- The app opens faster (the page-finding networks load in the background).
- Tap the shutter during the enhance animation to skip it and shoot the next page at once.

**New features**
- Draw / sign: pen, highlighter, typed text and a saved signature on any page (Review > Draw / sign).
- Save pages as pictures to the gallery, share as pictures, print.
- PDF options: paper size (fit, A4, Letter), page numbers, watermark.
- Home: sort (newest, oldest, name), list or grid, import photos from the gallery.
- Review: move a page earlier or later (the ⋮ menu). Editing a saved document opens on the page you were viewing.
- Settings > Storage: keep original photos on/off, space used, delete kept originals.

**Fixes**
- A scan that wasn't saved is no longer lost if Android closes the app: the home screen offers to continue it.
- Text recognition can no longer undo a page reorder or rotation made while it was running.
- The page finder no longer misses page edges that were split into pieces (fewer wrong outlines).
- "Rotating…" can't get stuck; the PDF password box is no longer hidden by the keyboard; a shared PDF is no longer deleted while another app is still reading it.


Kotlin + Jetpack Compose. No ads, no tracking, no account. Works offline.

## Features
- Own scanner (CameraX + OpenCV): live page outline, batch capture, tap to focus, flash, gallery import
- Edge detection v3 (91/100 on hard test photos, was 49/100): colour edges, line + blob candidates,
  every side checked for paper-inside/table-outside, real-corner check (rejects shadow lines),
  line re-fitting for exact corners; the crop sits ~1% inside the page so no table shows
- Perspective correction: tilted pages come out straight and rectangular
- Crop screen: drag corners or whole sides, magnifier while dragging, Auto detect, Whole photo
- Shadow removal v4: finds the bare paper first and estimates the lighting from it only, so shadows
  vanish while photos, black boxes, coloured headers and highlighter keep their look
- 7 filters: Auto, Magic color, Lighten, Grayscale, B&W, B&W soft, Original
- Sharper photos: the camera focuses on the page before each shot; pages saved at JPEG 93
- Pinch / double-tap zoom with full-detail images (review screen and saved pages)
- When edge detection isn't sure, the corner screen opens right after the photo
- Page-corner network (assets/docquad.onnx, MobileNetV3 + FPN): based on MakeACopy's DocQuadNet-256
  (Apache 2.0, trained on real photos), retrained on 24,000 realistic scenes (carpets, bedsheets, tiles,
  notebooks, receipts, fingers, shadows, dim light); combined with the line-based detector and the small
  page network (assets/pagenet.onnx). See licenses/NOTICE.txt
- Edit after saving: reopen a saved document's pages (original photos kept) to re-crop, change filters,
  rotate, delete or add pages; "Save changes" replaces the pages and keeps the name
- Auto-enhance animation after each photo, starting the instant the shutter fires (frozen picture,
  outline snaps onto the page, page lifts and straightens, scan line reveals the clean page)
- Sharp photos: focuses on the page, then waits until the phone is still (gyroscope) before shooting;
  warns "looks blurry – Retake" when a photo is shaken anyway
- Smooth live outline (colour + network, glides between positions, no flicker); auto-capture
- Never interrupts with the corner screen: doubtful pages get a "check the corners" note in review
- ID card mode (front + back on one A4 page at real size), searchable PDFs, password-protected PDFs
- Crash report: if the app ever closes unexpectedly, it offers to share the error details
- Photo quality setting: High (up to 13 MP, default) or Standard (5 MP, faster); small cached previews keep it smooth
- Smart file names from the first page's text, e.g. "Electricity bill – Oct 2026"
- One-tap **Save PDF** to Downloads with Original / Medium / Small sizes; share PDF
- Review opens on the photo you just took (page order and PDF order stay oldest → newest)
- Swipe between pages in review, in the corner editor and in saved documents
- Safe back navigation (no more blank white screen)
- Drag to reorder pages, rotate, delete, add pages; merge documents; import PDFs
- Offline text extraction (English/Latin), copy, share, fix by hand; search inside scans

## Build
Open this folder in Android Studio, wait for Gradle sync, then press Run with an Android phone connected.
See the step-by-step guide for details.

## Where things are
- `app/src/main/java/com/myscanner/app/scan/` — edge detection, shadow removal, filters, camera tools
- `app/src/main/java/com/myscanner/app/data/` — database, files, PDF building, OCR, smart names, settings
- `app/src/main/java/com/myscanner/app/ui/` — the screens
- `app/src/main/res/values/strings.xml` — the app name
- `app/build.gradle.kts` — app ID, version, libraries
