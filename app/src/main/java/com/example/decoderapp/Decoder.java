package com.example.decoderapp;

import android.content.Context;
import android.os.Environment;
import android.util.Log;
import java.io.*;
import java.nio.FloatBuffer;
import java.util.Collections;
import ai.onnxruntime.*;

public class Decoder {
    private OrtEnvironment env;
    private OrtSession session;

    public boolean initModel(Context ctx) {
        try {
            env = OrtEnvironment.getEnvironment();

            // assets/decoder_feature_ae.onnx  ← 刚导出的那个 (输入 1x128x4x4x4, 输出 1x4x128x128x128)
            InputStream is = ctx.getAssets().open("decoder_feature_ae.onnx");
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096]; int n;
            while ((n = is.read(buf)) > 0) baos.write(buf, 0, n);
            is.close();

            session = env.createSession(baos.toByteArray(), new OrtSession.SessionOptions());
            Log.i("DecoderApp", "✅ ONNX Loaded");
            for (NodeInfo info : session.getInputInfo().values()) {
                TensorInfo ti = (TensorInfo) info.getInfo();
                Log.i("DecoderApp", "Input: "+info.getName()+" shape="+java.util.Arrays.toString(ti.getShape()));
            }
            for (NodeInfo info : session.getOutputInfo().values()) {
                TensorInfo ti = (TensorInfo) info.getInfo();
                Log.i("DecoderApp", "Output: "+info.getName()+" shape="+java.util.Arrays.toString(ti.getShape()));
            }
            return true;
        } catch (Exception e) {
            Log.e("DecoderApp", "Model Load ERR: ", e);
            return false;
        }
    }

    /** 将 (1,4,4,4,128) 扁平 float[] 重排为 (1,128,4,4,4) 扁平 float[] */
    public float[] dhwc_to_ncdhw(float[] dhwc) {
        // 输入 dhwc 是按 (D,H,W,C) 顺序展开的 4*4*4*128 = 8192 个数
        // 输出 ncdhw 要按 (C,D,H,W) 展开
        float[] out = new float[128*4*4*4];
        int idx = 0;
        for (int d=0; d<4; d++)
            for (int h=0; h<4; h++)
                for (int w=0; w<4; w++)
                    for (int c=0; c<128; c++)
                        out[c*64 + d*16 + h*4 + w] = dhwc[idx++];
        return out;
    }

    /** Step 1: ONNX inference only */
    public static class DecodeResult {
        public float[] data;
        public double prepMs, inferMs, flattenMs, totalMs;

        public DecodeResult(float[] data, double prepMs, double inferMs, double flattenMs, double totalMs) {
            this.data = data;
            this.prepMs = prepMs;
            this.inferMs = inferMs;
            this.flattenMs = flattenMs;
            this.totalMs = totalMs;
        }

        public String getSummary() {
            return String.format(
                    "🧠 Detailed ONNX:\n" +
                            "Tensor prep: %.2f ms\n" +
                            "Infer: %.2f ms\n" +
                            "Flatten: %.2f ms\n" +
                            "Total: %.2f ms",
                    prepMs, inferMs, flattenMs, totalMs
            );
        }
    }

    public DecodeResult decodeFeatureGrid(float[] ncdhw) throws Exception {
        String inName = session.getInputNames().iterator().next();
        long[] shape = {1,128,4,4,4};

        long t0 = System.nanoTime();

        // 构建 tensor
        OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(ncdhw), shape);
        long tTensor = System.nanoTime();

        // 推理
        OrtSession.Result result = session.run(Collections.singletonMap(inName, input));
        long tInfer = System.nanoTime();

        // 扁平化
        float[][][][][] out5 = (float[][][][][]) result.get(0).getValue();
        float[] dhwc = new float[128*128*128*4];
        int i = 0;
        for (int z=0; z<128; z++)
            for (int h=0; h<128; h++)
                for (int w=0; w<128; w++)
                    for (int c=0; c<4; c++)
                        dhwc[i++] = out5[0][c][z][h][w];
        long tFlatten = System.nanoTime();

        input.close();
        result.close();

        double prepMs = (tTensor - t0) / 1e6;
        double inferMs = (tInfer - tTensor) / 1e6;
        double flattenMs = (tFlatten - tInfer) / 1e6;
        double totalMs = (tFlatten - t0) / 1e6;

        Log.i("DecoderApp", String.format(
                "🧠 ONNX detail — tensor: %.2f ms, infer: %.2f ms, flatten: %.2f ms, total: %.2f ms",
                prepMs, inferMs, flattenMs, totalMs));

        return new DecodeResult(dhwc, prepMs, inferMs, flattenMs, totalMs);
    }

    /** Step 2: Save to Download folder, DecoderAppResults/DecodedNPY subfolder */
    public File saveDecodedNpy(float[] decoded_dhwc) throws IOException {
        long t0 = System.currentTimeMillis();
        File dir = new File(Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS), "DecoderAppResults/DecodedNPY128");
        if (!dir.exists()) dir.mkdirs();
        File npy = new File(dir, "decoder_debug_" + System.currentTimeMillis() + ".npy");

        NpyWriter.writeNpy(npy.getAbsolutePath(), decoded_dhwc, new int[]{128,128,128,4});
        long t1 = System.currentTimeMillis();

        Log.i("DecoderApp", String.format("⏱ File write only: %.2fs", (t1 - t0)/1000.0));
        Log.i("DecoderApp", "✅ Saved: " + npy.getAbsolutePath());
        return npy;
    }
}
