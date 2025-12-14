package com.example.decoderapp

import android.Manifest
import android.content.Intent
import android.net.Uri
import android.os.Bundle
import android.os.Environment
import android.util.Log
import androidx.activity.ComponentActivity
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.Button
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.unit.dp
import androidx.core.app.ActivityCompat
import com.example.decoderapp.ui.theme.DecoderAppTheme
import java.io.File


class MainActivity : ComponentActivity() {
    private lateinit var decoder: Decoder
    /* 2025.11.12 Update: Add decoder64 */
    private lateinit var decoder64: Decoder64
    /* 2025.12.4 Update: Add Interpolator */
    private lateinit var interpolator: Interpolator

    @OptIn(ExperimentalMaterial3Api::class)
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        decoder = Decoder()
        decoder64 = Decoder64()
        /* 2025.12.4 Update: initialize interpolator*/
        interpolator = Interpolator()

        setContent {
            DecoderAppTheme {
                var fileUri by remember { mutableStateOf<Uri?>(null) }
                /* 2025.11.2 Update: Add Text to display status and add permission to read external storage */
                var statusMessage by remember { mutableStateOf("Ready.") }
                ActivityCompat.requestPermissions(
                    this,
                    arrayOf(Manifest.permission.WRITE_EXTERNAL_STORAGE),
                    1001
                )
                val dir =
                    Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS)
                val outDir = File(dir, "DecoderAppResults")
                outDir.mkdirs()

                val context = LocalContext.current

                /* 2025.11.3 Update: Substitute picker definition*/
//                val picker = rememberLauncherForActivityResult(
//                    ActivityResultContracts.GetContent()
//                ) { uri -> fileUri = uri }
                /* 如果直接 OpenDocument()，它只会展示媒体库（图片、音频…） */
                val pickerIntent = Intent(Intent.ACTION_GET_CONTENT).apply {
                    addCategory(Intent.CATEGORY_OPENABLE)
                    type = "*/*"
                }

                val picker = rememberLauncherForActivityResult(
                    ActivityResultContracts.StartActivityForResult()
                ) { result ->
                    val uri = result.data?.data ?: return@rememberLauncherForActivityResult
                    fileUri = uri

                    /* 2025.11.13 Update: Add file name display*/
                    val name = FileUtil.getFileName(this@MainActivity, uri)
                    statusMessage = "📄 Selected: ${name ?: "Unknown file"}"
                }

                /* 2025.12.4 Update: Add picker A and B for interpolation */
                var filelatentAUri by remember { mutableStateOf<Uri?>(null) }
                var filelatentBUri by remember { mutableStateOf<Uri?>(null) }
                val latentAUri = rememberLauncherForActivityResult(
                    ActivityResultContracts.StartActivityForResult()
                ) { result ->
                    val uri = result.data?.data ?: return@rememberLauncherForActivityResult
                    filelatentAUri = uri

                    val name = FileUtil.getFileName(this@MainActivity, uri)
                    statusMessage = "📄 Selected: ${name ?: "Unknown file"}"
                }

                val latentBUri = rememberLauncherForActivityResult(
                    ActivityResultContracts.StartActivityForResult()
                ) { result ->
                    val uri = result.data?.data ?: return@rememberLauncherForActivityResult
                    filelatentBUri = uri

                    val name = FileUtil.getFileName(this@MainActivity, uri)
                    statusMessage = "📄 Selected: ${name ?: "Unknown file"}"
                }

                /* 2025.11.13 Update: Add scroll bar*/
                val scrollState = rememberScrollState()
                Scaffold(
                    topBar = { TopAppBar(title = { Text("Decoder App") }) }
                ) { pad ->
                    Column(
                        Modifier
                            .padding(pad)
                            .padding(16.dp)
                            .verticalScroll(scrollState)   //2025.11.13 Update: Add scroll bar
                            .fillMaxWidth(),
                        horizontalAlignment = Alignment.CenterHorizontally
                    ) {
                        Button(onClick = {
                            picker.launch(pickerIntent)
                        }) {
                            Text("Choose latent file (.npy or .bin)")
                        }

                        // Status message between buttons
                        Spacer(Modifier.height(12.dp))
                        Text(statusMessage)

                        Spacer(Modifier.height(12.dp))
                        Button(onClick = {
                            Thread {
                                try {
                                    if (!decoder.initModel(this@MainActivity)) {
                                        runOnUiThread { statusMessage = "❌ Fail to load model" }
                                        return@Thread
                                    }

                                    val uri = fileUri ?: run {
                                        runOnUiThread { statusMessage = "❌ Please choose a file" }
                                        return@Thread
                                    }

                                    val t0 = System.nanoTime()
                                    val dhwc = FileUtil.loadEmbedNPY_DHWC(this@MainActivity, uri)
                                    val t1 = System.nanoTime()

                                    val ncdhw = decoder.dhwc_to_ncdhw(dhwc)
                                    val t2 = System.nanoTime()

                                    /**  context 是 Android 的环境对象（比如 Activity、Service、Application）。
                                        它代表当前运行的环境，可以访问文件系统、资源、应用目录等。
                                        在 Activity 里，this 或 this@MainActivity 就是一个 Context。
                                        但在普通类（比如 Decoder.java）里没有自动的 Context，所以我们要从外面传进去。
                                    You just need to pass the current Activity context into runDecodeFromNCDHW().
                                    This allows the decoder to safely access the app’s private directory (getExternalFilesDir)
                                    without global references or permission issues.
                                     */

                                    /*  2025.11.11 Update: Separate inference and file writing. */
                                    //  decoder.runDecodeFromNCDHW(this@MainActivity, ncdhw)
                                    val result = decoder.decodeFeatureGrid(ncdhw)
                                    val decoded_dhwc = result.data

                                    val t3 = System.nanoTime()

                                    val saved = decoder.saveDecodedNpy(decoded_dhwc)

                                    val t4 = System.nanoTime()


                                    val loadMs = (t1 - t0) / 1e6
                                    val reshapeMs = (t2 - t1) / 1e6
                                    val inferMs = (t3 - t2) / 1e6
                                    val filewritingMs = (t4 - t3) / 1e6
                                    val totalMs = (t4 - t0) / 1e6

                                    runOnUiThread {
                                        statusMessage = """
✅ Decoding Finished!
📥 Load NPY: ${"%.2f".format(loadMs)} ms
🔄 Shape Convert: ${"%.2f".format(reshapeMs)} ms
🧠 Decode ONNX: ${"%.2f".format(inferMs)} ms
${result.getSummary()}
💾 Save decoded NPY: ${"%.2f".format(filewritingMs)} ms
⏱ Total: ${"%.2f".format(totalMs)} ms
Saved: ${saved.name}
""".trimIndent()
                                    }
                                } catch (e: Exception) {
                                    Log.e("DecoderApp", "Error: $e", e)
                                    runOnUiThread { statusMessage = "❌ ERROR (128): ${e.message ?: "See Logcat"}"}
                                }
                            }.start()
                        }) {
                            Text("Run Decoder (128)")
                        }

                        /* 2025.11.12 Update: Add decode 64 button, use new model.
                           2025.11.13 Update: Add Load NPY time, shape convert time, infer time, save decoded NPY time.
                        */

                        Spacer(Modifier.height(12.dp))

                        Button(onClick = {
                            Thread {
                                try {
                                    val uri = fileUri ?: run {
                                        runOnUiThread { statusMessage = "❌ Please choose a latent file" }
                                        return@Thread
                                    }

                                    val t0 = System.nanoTime()
                                    val dhwc = FileUtil.loadEmbedNPY_DHWC_64(this@MainActivity, uri)
                                    /* 2025.11.13 Update: Not allowed to use 128-channel latent */
                                    if (dhwc.size != 4096) {
                                        runOnUiThread {
                                            statusMessage = "❌ Wrong latent size: expected 4×4×4×64 (4096 floats), got ${dhwc.size}"
                                        }
                                        return@Thread
                                    }

                                    val t1 = System.nanoTime()

                                    val decoder64 = Decoder64()
                                    if (!decoder64.initModel(this@MainActivity)) {
                                        runOnUiThread { statusMessage = "❌ Fail to load 64-ch model" }
                                        return@Thread
                                    }

                                    val ncdhw64 = decoder64.dhwc_to_ncdhw_64(dhwc)
                                    val t2 = System.nanoTime()

                                    val result = decoder64.decodeFeatureGrid64(ncdhw64)
                                    val decoded_dhwc = result.data
                                    val t3 = System.nanoTime()

                                    val saved = decoder64.saveDecodedNpy64(decoded_dhwc)
                                    val t4 = System.nanoTime()

                                    val loadMs = (t1 - t0) / 1e6
                                    val convertMs = (t2 - t1) / 1e6
                                    val inferMs = (t3 - t2) / 1e6
                                    val saveMs = (t4 - t3) / 1e6
                                    val totalMs = (t4 - t0) / 1e6

                                    runOnUiThread {
                                        statusMessage = """
✅ 64-Ch Decoding Finished!
📥 Load Latent NPY: ${"%.2f".format(loadMs)} ms
🔄 Shape Convert: ${"%.2f".format(convertMs)} ms
🧠 Decode ONNX: ${"%.2f".format(inferMs)} ms
${result.getSummary()}
💾 Save decoded NPY: ${"%.2f".format(saveMs)} ms
⏱ Total: ${"%.2f".format(totalMs)} ms
Saved: ${saved.name}
""".trimIndent()
                                    }
                                } catch (e: Exception) {
                                    Log.e("DecoderApp", "ERR64", e)
                                    runOnUiThread {
                                        statusMessage = "❌ ERROR (64): ${e.message ?: "See Logcat"}"
                                    }
                                }
                            }.start()
                        }) {
                            Text("Run Decoder (64)")
                        }

                        /* 2025.12.4 Update: Add Interpolation */
                        Spacer(Modifier.height(12.dp))
                        Button(onClick = {
                            latentAUri.launch(pickerIntent)
                        }) {
                            Text("Choose first latent file (.npy or .bin)")
                        }
                        Button(onClick = {
                            latentBUri.launch(pickerIntent)
                        }) {
                            Text("Choose second latent file (.npy or .bin)")
                        }

                        Spacer(Modifier.height(12.dp))
                        // Edit testing part, ensure Decoder is initialized
                        Button(onClick = {
                            Thread {
                                try {
                                    val uriA = filelatentAUri ?: return@Thread
                                    val uriB = filelatentBUri ?: return@Thread

                                    Log.d("Interpolation", "Testing started...")

                                    // 1. Ensure Decoder is initialized
                                    if (!decoder.initModel(this@MainActivity)) {
                                        val initSuccess = decoder.initModel(this@MainActivity)
                                        if (!initSuccess) {
                                            runOnUiThread { statusMessage = "❌ Decoder Initialization failed" }
                                            return@Thread
                                        }
                                    }

                                    // 2. Load embed data

                                    // 2025.12.11 Update: add A/B load time
                                    val tStart = System.nanoTime()
                                    val tLoad0 = System.nanoTime()
                                    val A_data = FileUtil.loadEmbedNPY_DHWC(this@MainActivity, uriA)
                                    val B_data = FileUtil.loadEmbedNPY_DHWC(this@MainActivity, uriB)
                                    val tLoad1 = System.nanoTime()

                                    if (A_data.size != 8192 || B_data.size != 8192) {
                                        runOnUiThread {
                                            statusMessage = "need embed data of size 8192 floats"
                                        }
                                        return@Thread
                                    }

                                    // 3. Initialize interpolator

                                    // 2025.12.11 Add init interpolator time
                                    val tInterpInit0 = System.nanoTime()
                                    val interpolator = Interpolator()
                                    if (!interpolator.initModel(this@MainActivity)) {
                                        runOnUiThread { statusMessage = "Interpolator model load failed" }
                                        return@Thread
                                    }
                                    val tInterpInit1 = System.nanoTime()

                                    // 4. Run interpolator
                                    // 2025.12.11 Add infer time
                                    val tInfer0 = System.nanoTime()
                                    val frames = interpolator.interpolateSimple(A_data, B_data)
                                    val tInfer1 = System.nanoTime()
                                    interpolator.close()

                                    if (frames.isEmpty()) {
                                        runOnUiThread { statusMessage = "Interpolation Failed" }
                                        return@Thread
                                    }

                                    // 5. Save files
                                    // 2025.12.11 Add saving time
                                    val tSave0 = System.nanoTime()
                                    val outDir = File(
                                        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                                        "DecoderAppResults/InterpolationTest"
                                    )
                                    outDir.mkdirs()

                                    /* 2025.12.11 Update: Improved interpolation frame id */
                                    for ((index, frame) in frames.withIndex()) {
                                        val frameId = index + 1     //  0→1, 1→2, 2→3

                                        val npyFile = File(outDir, "interpolation_frame_$frameId.npy")
                                        NpyWriter.writeNpy(npyFile.absolutePath, frame, intArrayOf(128, 4, 4, 4))

                                        Log.d("Interpolation", "Saved: ${npyFile.name}")
                                    }
                                    val tSave1 = System.nanoTime()

                                    val tEnd = System.nanoTime()

                                    //----------------------------
                                    //  Convert to ms
                                    //----------------------------
                                    val loadMs = (tLoad1 - tLoad0) / 1e6
                                    val initMs = (tInterpInit1 - tInterpInit0) / 1e6
                                    val inferMs = (tInfer1 - tInfer0) / 1e6
                                    val saveMs = (tSave1 - tSave0) / 1e6
                                    val totalMs = (tEnd - tStart) / 1e6

                                    runOnUiThread {
                                        statusMessage = """
                    ✅ Interpolation Completed!

                    📥 Load A/B embed: ${"%.2f".format(loadMs)} ms
                    🔧 Init Interpolator: ${"%.2f".format(initMs)} ms
                    🧠 Interpolate ONNX: ${"%.2f".format(inferMs)} ms
                    💾 Save 3 frames: ${"%.2f".format(saveMs)} ms

                    ⏱ Total: ${"%.2f".format(totalMs)} ms

                    Files saved to:
                    ${outDir.absolutePath}

                    Next Steps:
                    1. Use "Run Decoder (128)" on interpolation_frame_X.npy
                    2. Then "Convert Decoded NPY (128) to OBJ"
                """.trimIndent()
                                    }

                                } catch (e: Exception) {
                                    Log.e("Interpolation", "error", e)
                                    runOnUiThread {
                                        statusMessage = "error: ${e.message}"
                                    }
                                }
                            }.start()
                        }) {
                            Text("Run Interpolator (128)")
                        }

                        /* 2025.12.12 Update: Add 64-ch interpolation */
                        Spacer(Modifier.height(12.dp))
                        // Edit testing part, ensure Decoder is initialized
                        Button(onClick = {
                            Thread {
                                try {
                                    val uriA = filelatentAUri ?: return@Thread
                                    val uriB = filelatentBUri ?: return@Thread

                                    val tStart = System.nanoTime()
                                    val tLoad0 = System.nanoTime()
                                    // 2025.12.12 Update add A/B load time
                                    val A_data = FileUtil.loadEmbedNPY_DHWC_64(this@MainActivity, uriA)
                                    val B_data = FileUtil.loadEmbedNPY_DHWC_64(this@MainActivity, uriB)
                                    val tLoad1 = System.nanoTime()

                                    if (A_data.size != 4096 || B_data.size != 4096) {
                                        runOnUiThread { statusMessage = "Need 64 channel embed（4096 floats）" }
                                        return@Thread
                                    }

                                    val tInterpInit0 = System.nanoTime()
                                    val interpolator = Interpolator64()
                                    if (!interpolator.initModel(this@MainActivity)) {
                                        runOnUiThread { statusMessage = "Interpolator 64 load failed" }
                                        return@Thread
                                    }
                                    val tInterpInit1 = System.nanoTime()

                                    // 4. Run interpolator
                                    // 2025.12.11 Add infer time
                                    val tInfer0 = System.nanoTime()
                                    val frames = interpolator.interpolateSimple(A_data, B_data)
                                    val tInfer1 = System.nanoTime()
                                    interpolator.close()

                                    if (frames.isEmpty()) {
                                        runOnUiThread { statusMessage = "Interpolation Failed" }
                                        return@Thread
                                    }

                                    // 5. Save files
                                    // 2025.12.11 Add saving time
                                    val tSave0 = System.nanoTime()
                                    val outDir = File(
                                        Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                                        "DecoderAppResults/InterpolationTest"
                                    )
                                    outDir.mkdirs()

                                    /* 2025.12.11 Update: Improved interpolation frame id */
                                    for ((index, frame) in frames.withIndex()) {
                                        val frameId = index + 1     //  0→1, 1→2, 2→3

                                        val npyFile = File(outDir, "interpolation_frame_$frameId.npy")
                                        NpyWriter.writeNpy(npyFile.absolutePath, frame, intArrayOf(64, 4, 4, 4))

                                        Log.d("Interpolation", "Saved: ${npyFile.name}")
                                    }
                                    val tSave1 = System.nanoTime()

                                    val tEnd = System.nanoTime()

                                    //----------------------------
                                    //  Convert to ms
                                    //----------------------------
                                    val loadMs = (tLoad1 - tLoad0) / 1e6
                                    val initMs = (tInterpInit1 - tInterpInit0) / 1e6
                                    val inferMs = (tInfer1 - tInfer0) / 1e6
                                    val saveMs = (tSave1 - tSave0) / 1e6
                                    val totalMs = (tEnd - tStart) / 1e6

                                    runOnUiThread {
                                        statusMessage = """
                    ✅ Interpolation Completed!

                    📥 Load A/B embed: ${"%.2f".format(loadMs)} ms
                    🔧 Init Interpolator: ${"%.2f".format(initMs)} ms
                    🧠 Interpolate ONNX: ${"%.2f".format(inferMs)} ms
                    💾 Save 3 frames: ${"%.2f".format(saveMs)} ms

                    ⏱ Total: ${"%.2f".format(totalMs)} ms

                    Files saved to:
                    ${outDir.absolutePath}

                    Next Steps:
                    1. Use "Run Decoder (64)" on interpolation_frame_X.npy
                    2. Then "Convert Decoded NPY (64) to OBJ"
                """.trimIndent()
                                    }

                                } catch (e: Exception) {
                                    Log.e("Interpolation", "error", e)
                                    runOnUiThread {
                                        statusMessage = "error: ${e.message}"
                                    }
                                }
                            }.start()
                        }) {
                            Text("Run Interpolator (64)")
                        }

                        /* 2025.11.5 Update: Add conversion buttons*/
                        val pickerIntent2 = Intent(Intent.ACTION_GET_CONTENT).apply {
                            addCategory(Intent.CATEGORY_OPENABLE)
                            type = "*/*"
                        }
                        Spacer(Modifier.height(12.dp))
                        Button(
                            onClick = { picker.launch(pickerIntent2) }
                        ) { Text("Select Decoded NPY") }

                        Spacer(Modifier.height(12.dp))
                        Button(
                            onClick = {
                                Thread {
                                    try {
                                        val uri = fileUri ?: run {
                                            runOnUiThread { statusMessage = "❌Please choose decoded npy" }
                                            return@Thread
                                        }

                                        /* 2025.11.20 Update: Sync UI for old OBJ conversion */
                                        val t0 = System.nanoTime()
                                        val raw = FileUtil.loadNpyFloat(this@MainActivity, uri)
                                        val t1 = System.nanoTime()

                                        val mc = MarchingCubes.fromDecoderOutput(raw)
                                        val mesh = mc.generate()
                                        val t2 = System.nanoTime()

                                        val objFile = MarchingCubes.saveObj(mesh)
                                        val t3 = System.nanoTime()

                                        val loadMs = (t1 - t0) / 1e6
                                        val mcMs = (t2 - t1) / 1e6
                                        val saveMs = (t3 - t2) / 1e6
                                        val totalMs = (t3 - t0) / 1e6

                                        val vCount = mesh.vertices.size / 3
                                        val fCount = mesh.faces.size / 3

                                        runOnUiThread {
                                            statusMessage = """
                                                                ✅ OBJ Generated (128³ TSDF)
                                                                📥 Load decoded NPY: ${"%.2f".format(loadMs)} ms
                                                                🧱 MarchingCubes: ${"%.2f".format(mcMs)} ms
                                                                💾 Save OBJ: ${"%.2f".format(saveMs)} ms
                                                                🔢 Vertices: $vCount
                                                                🔺 Faces: $fCount
                                                                ⏱ Total: ${"%.2f".format(totalMs)} ms
                                                                File: ${objFile.name}
                                                            """.trimIndent()
                                        }
                                    } catch (e: Exception) {
                                        Log.e("DecoderApp","ERR: $e",e)
                                        runOnUiThread { statusMessage = "❌ Failed to convert to OBJ" }
                                    }
                                }.start()
                            }
                        ) { Text("Convert Decoded NPY (128) to OBJ") }

                        Spacer(Modifier.height(12.dp))
                        /*  2025.11.12 Update: Add button to convert 64-ch NPY to OBJ*/
                        /* 2025.11.13 Update: Add vertices and faces number display*/
                        Button(onClick = {
                            Thread {
                                try {
                                    val uri = fileUri ?: run {
                                        runOnUiThread { statusMessage = "❌ Please choose a decoded NPY (4MB)" }
                                        return@Thread
                                    }

                                    val t0 = System.nanoTime()
                                    val raw = FileUtil.loadNpyFloat(this@MainActivity, uri)
                                    val t1 = System.nanoTime()

                                    val mc = MarchingCubes64.fromDecoderOutput64(raw)
                                    val mesh = mc.generate()
                                    val t2 = System.nanoTime()

                                    val objFile = MarchingCubes64.saveObj(mesh)
                                    val t3 = System.nanoTime()

                                    val loadMs = (t1 - t0) / 1e6
                                    val mcMs = (t2 - t1) / 1e6
                                    val saveMs = (t3 - t2) / 1e6
                                    val totalMs = (t3 - t0) / 1e6

                                    val vCount = mesh.vertices.size / 3
                                    val fCount = mesh.faces.size / 3

                                    runOnUiThread {
                                        statusMessage = """
                    ✅ OBJ Generated (64³ TSDF)
                    📥 Load decoded NPY: ${"%.2f".format(loadMs)} ms
                    🧱 MarchingCubes: ${"%.2f".format(mcMs)} ms
                    💾 Save OBJ: ${"%.2f".format(saveMs)} ms
                    🔢 Vertices: $vCount
                    🔺 Faces: $fCount
                    ⏱ Total: ${"%.2f".format(totalMs)} ms
                    File: ${objFile.name}
                    You can view OBJ files in a 3D Model Viewer software.
                """.trimIndent()
                                    }

                                } catch (e: Exception) {
                                    Log.e("DecoderApp", "OBJ_ERR64", e)
                                    runOnUiThread { statusMessage = "❌ Error converting decoded NPY → OBJ" }
                                }
                            }.start()
                        }) {
                            Text("Convert Decoded NPY (64) to OBJ")
                        }
                        /* 2025.12.12 Update: Add annotation */
                        Spacer(Modifier.height(12.dp))
                        var annotation by remember { mutableStateOf("This is the final project of EECE5512 Networked XR Systems course, Fall 2025 Semester, Northeastern University. Developed by Jingming Cheng from Sep 30th 2025 to Dec 12th 2025 in Boston, MA. Contact: cheng.jingm@northeastern.edu or jingmingcheng@outlook.com") }
                        Text(annotation)
                    }
                }
            }
        }
    }
}
