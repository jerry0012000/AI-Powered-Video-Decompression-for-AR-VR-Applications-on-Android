package com.example.decoderapp

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

class Interpolator64 {
    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null

    fun initModel(context: Context): Boolean {
        return try {
            env = OrtEnvironment.getEnvironment()

            // 模型名改成你导出的 64 模型
            val modelFile = File(context.filesDir, "transformer_compressed_64.onnx")

            if (!modelFile.exists()) {
                context.assets.open("transformer_compressed_64.onnx").use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }

            session = env!!.createSession(modelFile.path)
            Log.d("Interpolator64", "Model load success: transformer_compressed_64.onnx")
            true
        } catch (e: Exception) {
            Log.e("Interpolator64", "Model load failed", e)
            false
        }
    }

    /**
     * 简单插值 (64-dim)
     * 输入: [1, 64, 4, 4, 4]
     * 输出: List< FloatArray >, 每帧4096 floats
     */
    fun interpolateSimple(embedA: FloatArray, embedB: FloatArray): List<FloatArray> {
        val session = session ?: throw IllegalStateException("Model not initialized")

        return try {

            // -----------------------------
            // 1) 输入 tensor 尺寸 (64 instead of 128)
            // -----------------------------
            val shape = longArrayOf(1, 64, 4, 4, 4)

            val tensorA = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(embedA), shape)
            val tensorB = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(embedB), shape)

            // -----------------------------
            // 2) d_codes = [1, 3, 32]
            // -----------------------------
            val dCodes = FloatArray(1 * 3 * 32) { (Math.random().toFloat() - 0.5f) * 0.1f }
            val dCodesTensor = OnnxTensor.createTensor(
                env!!,
                FloatBuffer.wrap(dCodes),
                longArrayOf(1, 3, 32)
            )

            // -----------------------------
            // 3) Run inference
            // -----------------------------
            val results = session.run(
                mapOf(
                    "embed_A" to tensorA,
                    "embed_B" to tensorB,
                    "d_codes" to dCodesTensor
                )
            )

            val output = results[0] as OnnxTensor
            val raw = output.floatBuffer.array()

            Log.d("Interpolator64", "Model output size = ${raw.size}")

            // -----------------------------
            // 4) 转换格式返回 3 帧
            // shape = [3, 64, 4, 4, 4] → 每帧4096 floats
            // -----------------------------
            convertForDecoder(raw, channels = 64)

        } catch (e: Exception) {
            Log.e("Interpolator64", "Interpolation failed", e)
            throw e
        }
    }

    private fun convertForDecoder(data: FloatArray, channels: Int): List<FloatArray> {
        val frames = 3
        val valuesPerFrame = channels * 4 * 4 * 4  // 64*4*4*4 = 4096

        val expected = frames * valuesPerFrame
        if (data.size != expected) {
            Log.e("Interpolator64",
                "Output size mismatch! expectedResult=$expected, but got data.size=${data.size}"
            )
        }

        val result = mutableListOf<FloatArray>()

        for (i in 0 until frames) {
            val start = i * valuesPerFrame
            val end = start + valuesPerFrame
            if (end > data.size) break

            val frame = data.copyOfRange(start, end)
            Log.d("Interpolator64", "Frame $i size = ${frame.size}")

            result.add(frame)
        }
        return result
    }

    fun close() {
        session?.close()
    }
}
