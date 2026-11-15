plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("org.jetbrains.kotlin.plugin.compose")
    id("com.google.devtools.ksp")
}

android {
    namespace = "com.myscanner.app"
    compileSdk = 35

    defaultConfig {
        // This is your app's unique ID on Google Play. Change it once, before your
        // first upload (for example "com.yourname.scanner"), and never again.
        applicationId = "com.myscanner.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 13
        versionName = "8.3"

        // Only real phone CPUs. Keeps the app much smaller (OpenCV is large).
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a")
        }
    }

    lint { checkReleaseBuilds = false; abortOnError = false }

    buildTypes {
        // "release" is the fast version of the app (Android runs it fully optimised; "debug" is
        // 2-5x slower). It is signed with the same debug key, so Android Studio can install it
        // directly: Build > Select Build Variant > release.
        release {
            isMinifyEnabled = false
            isDebuggable = false
            signingConfig = signingConfigs.getByName("debug")
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlinOptions {
        jvmTarget = "17"
    }
    buildFeatures {
        compose = true
    }

    // Tests that run on the PC (no phone needed). The image tests need a desktop OpenCV library:
    // they are skipped unless the OPENCV_JVM_LIB environment variable points to its folder.
    testOptions {
        unitTests {
            isIncludeAndroidResources = true
            all { test ->
                test.maxHeapSize = "3g"
                System.getenv("OPENCV_JVM_LIB")?.let { test.jvmArgs("-Djava.library.path=$it") }
                System.getenv("ROBOLECTRIC_REPO")?.let { test.systemProperty("robolectric.dependency.repo.url", it) }
                test.systemProperty("scanner.testdata", System.getenv("SCANNER_TESTDATA") ?: "")
                test.testLogging { showStandardStreams = true }
            }
        }
    }
}

dependencies {
    // Jetpack Compose (the UI toolkit)
    val composeBom = platform("androidx.compose:compose-bom:2024.12.01")
    implementation(composeBom)
    implementation("androidx.compose.ui:ui")
    implementation("androidx.compose.material3:material3")
    implementation("androidx.compose.material:material-icons-extended")
    implementation("androidx.compose.ui:ui-tooling-preview")
    debugImplementation("androidx.compose.ui:ui-tooling")

    implementation("androidx.core:core-ktx:1.15.0")
    implementation("androidx.activity:activity-compose:1.9.3")
    implementation("androidx.navigation:navigation-compose:2.8.5")
    implementation("androidx.lifecycle:lifecycle-viewmodel-compose:2.8.7")
    implementation("androidx.lifecycle:lifecycle-runtime-compose:2.8.7")

    // Saved settings
    implementation("androidx.datastore:datastore-preferences:1.1.1")

    // Database
    implementation("androidx.room:room-runtime:2.6.1")
    implementation("androidx.room:room-ktx:2.6.1")
    ksp("androidx.room:room-compiler:2.6.1")

    // Our own camera (CameraX)
    val camerax = "1.4.0"
    implementation("androidx.camera:camera-core:$camerax")
    implementation("androidx.camera:camera-camera2:$camerax")
    implementation("androidx.camera:camera-lifecycle:$camerax")
    implementation("androidx.camera:camera-view:$camerax")

    // OpenCV: edge detection, perspective correction, shadow removal, filters
    implementation("org.opencv:opencv:4.10.0")
    implementation("androidx.exifinterface:exifinterface:1.3.7")

    // Google ML Kit offline text recognition (OCR + smart file names)
    implementation("com.google.mlkit:text-recognition:16.0.1")
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-play-services:1.9.0")

    // PDF building with real JPEG compression, and merging
    implementation("com.tom-roush:pdfbox-android:2.0.27.0")

    // Image loading for thumbnails
    implementation("io.coil-kt:coil-compose:2.7.0")

    // Drag-to-reorder pages
    implementation("sh.calvin.reorderable:reorderable:2.4.3")

    // Tests (run on the PC with Robolectric)
    testImplementation("junit:junit:4.13.2")
    testImplementation("org.robolectric:robolectric:4.14.1")
    testImplementation("androidx.test:core-ktx:1.6.1")
    testImplementation("androidx.test.ext:junit:1.2.1")
    testImplementation("org.jetbrains.kotlinx:kotlinx-coroutines-test:1.9.0")
    testImplementation(composeBom)
    testImplementation("androidx.compose.ui:ui-test-junit4")
    testImplementation("androidx.test.espresso:espresso-core:3.6.1")
    debugImplementation("androidx.compose.ui:ui-test-manifest")
}
