// Interpolator.kt - 简化版
package com.example.decoderapp

import android.content.Context
import android.util.Log
import ai.onnxruntime.*
import java.io.File
import java.io.FileOutputStream
import java.nio.FloatBuffer

class Interpolator {
    private var session: OrtSession? = null
    private var env: OrtEnvironment? = null

    fun initModel(context: Context): Boolean {
        return try {
            env = OrtEnvironment.getEnvironment()
            val modelFile = File(context.filesDir, "transformer_compressed.onnx")

            if (!modelFile.exists()) {
                context.assets.open("transformer_compressed.onnx").use { input ->
                    FileOutputStream(modelFile).use { output ->
                        input.copyTo(output)
                    }
                }
            }

            session = env!!.createSession(modelFile.path)
            Log.d("Interpolator", "Model load success: transformer_compressed.onnx")
            true
        } catch (e: Exception) {
            Log.e("Interpolator", "Model load failed", e)
            false
        }
    }

    /**
     * 简单插值：输入两个embed特征，输出3个中间特征
     * 输入: [1, 128, 4, 4, 4]
     * 输出: 3个 [128, 4, 4, 4] 数组
     */
    fun interpolateSimple(embedA: FloatArray, embedB: FloatArray): List<FloatArray> {
        val session = session ?: throw IllegalStateException("Model not initialized")

        return try {
            // 准备输入张量 [1, 128, 4, 4, 4]
            val shape = longArrayOf(1, 128, 4, 4, 4)
            val tensorA = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(embedA), shape)
            val tensorB = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(embedB), shape)

            // d_codes: [1, 3, 32] 使用随机值
            val dCodes = FloatArray(1 * 3 * 32) { (Math.random().toFloat() - 0.5f) * 0.1f }
            val dCodesTensor = OnnxTensor.createTensor(env!!, FloatBuffer.wrap(dCodes), longArrayOf(1, 3, 32))

            // 运行推理
            val inputs = mapOf(
                "embed_A" to tensorA,
                "embed_B" to tensorB,
                "d_codes" to dCodesTensor
            )

            val results = session.run(inputs)
            val outputTensor = results[0] as OnnxTensor
            val outputData = outputTensor.floatBuffer.array()

            Log.d("Interpolator", "Raw output size: ${outputData.size} floats")

            // 关键：转换为Decoder可用的格式
            convertForDecoder(outputData)

        } catch (e: Exception) {
            Log.e("Interpolator", "Interpolation failed", e)
            throw e
        }
    }

    private fun convertForDecoder(data: FloatArray): List<FloatArray> {
        // 假设 ONNX Runtime 输出的 FloatArray 内存布局
        // 与其张量形状 [3, 128, 4, 4, 4] 的逻辑顺序完全一致。
        // 即：第一个 8192 个浮点数是第0帧，接着的 8192 个是第1帧，以此类推。

        val frames = 3
        val valuesPerFrame = 128 * 4 * 4 * 4 // 等于 8192

        // 检查总数据量是否符合预期
        val expectedTotalSize = frames * valuesPerFrame
        if (data.size != expectedTotalSize) {
            Log.e("Interpolator", "Output size mismatch! expectedResult=$expectedTotalSize, but got data.size=${data.size}")
            // 即使大小不匹配，也尝试处理，但记录严重错误
        }

        val result = mutableListOf<FloatArray>()

        for (f in 0 until frames) {
            val startIndex = f * valuesPerFrame
            val endIndex = startIndex + valuesPerFrame

            // 安全检查，防止数据大小不足导致崩溃
            if (endIndex > data.size) {
                Log.e("Interpolator", "Data not enough for $f th frame, pipeline stopped")
                break // 中止循环
            }

            // 直接、简单、可靠地按顺序分割数组
            val frameData = data.copyOfRange(startIndex, endIndex)
            result.add(frameData)

            Log.d("Interpolator", "Frame $f extracted finished, size: ${frameData.size}")
        }

        return result
    }


    fun close() {
        session?.close()
    }
}