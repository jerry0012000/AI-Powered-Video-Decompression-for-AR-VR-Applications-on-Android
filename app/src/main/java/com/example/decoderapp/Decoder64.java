package com.example.decoderapp;

import android.content.Context;
import android.os.Environment;
import android.util.Log;
import java.io.*;
import java.nio.FloatBuffer;
import java.util.Collections;
import ai.onnxruntime.*;

/**
 * Feature Decoder for the 64-channel model (resolution 4×4×4×64 -> 64×64×64×4 output)
 */
public class Decoder64 {

    private OrtEnvironment env;
    private OrtSession session;

    /** -----------------------------
     *  Load ONNX model (assets/decoder_feature64.onnx)
     *  ----------------------------- */
    public boolean initModel(Context ctx) {
        try {
            env = OrtEnvironment.getEnvironment();

            InputStream is = ctx.getAssets().open("decoder_feature64.onnx");
            ByteArrayOutputStream baos = new ByteArrayOutputStream();
            byte[] buf = new byte[4096];
            int n;
            while ((n = is.read(buf)) > 0) baos.write(buf, 0, n);
            is.close();

            session = env.createSession(baos.toByteArray(), new OrtSession.SessionOptions());
            Log.i("DecoderApp", "✅ Decoder64 ONNX Loaded");

            return true;
        } catch (Exception e) {
            Log.e("DecoderApp", "❌ Decoder64 Load ERR", e);
            return false;
        }
    }



    /** -----------------------------
     *  (4,4,4,64) DHWC → (1,64,4,4,4) NCDHW
     *  ----------------------------- */
    public float[] dhwc_to_ncdhw_64(float[] dhwc) {
        // 输入扁平：4*4*4*64 = 4096 floats
        // 输出扁平：64*4*4*4 = 4096 floats
        float[] out = new float[64 * 4 * 4 * 4];

        int idx = 0;
        for (int d = 0; d < 4; d++)
            for (int h = 0; h < 4; h++)
                for (int w = 0; w < 4; w++)
                    for (int c = 0; c < 64; c++)
                        out[c * 64 + d * 16 + h * 4 + w] = dhwc[idx++];

        return out;
    }



    /** -----------------------------
     *  Timing struct
     *  ----------------------------- */
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
                    "🧠 Detailed ONNX (64-ch):\n" +
                            "Tensor prep: %.2f ms\n" +
                            "Infer: %.2f ms\n" +
                            "Flatten: %.2f ms\n" +
                            "Total: %.2f ms",
                    prepMs, inferMs, flattenMs, totalMs
            );
        }
    }



    /** -----------------------------
     *  Run ONNX: input (1,64,4,4,4) → output (1,4,64,64,64)
     *  ----------------------------- */
    public DecodeResult decodeFeatureGrid64(float[] ncdhw) throws Exception {
        String inName = session.getInputNames().iterator().next();
        long[] shape = {1, 64, 4, 4, 4};

        long t0 = System.nanoTime();

        // tensor
        OnnxTensor input = OnnxTensor.createTensor(env, FloatBuffer.wrap(ncdhw), shape);
        long tTensor = System.nanoTime();

        // inference
        OrtSession.Result result = session.run(Collections.singletonMap(inName, input));
        long tInfer = System.nanoTime();

        // flatten (DHWC)
        float[][][][][] out5 = (float[][][][][]) result.get(0).getValue();
        float[] dhwc = new float[64 * 64 * 64 * 4];
        int i = 0;

        for (int z = 0; z < 64; z++)
            for (int y = 0; y < 64; y++)
                for (int x = 0; x < 64; x++)
                    for (int c = 0; c < 4; c++)
                        dhwc[i++] = out5[0][c][z][y][x];

        long tFlatten = System.nanoTime();

        input.close();
        result.close();

        // timing
        return new DecodeResult(
                dhwc,
                (tTensor - t0) / 1e6,
                (tInfer - tTensor) / 1e6,
                (tFlatten - tInfer) / 1e6,
                (tFlatten - t0) / 1e6
        );
    }



    /** -----------------------------
     *  Save decoded NPY (64×64×64×4)
     *  ----------------------------- */
    public File saveDecodedNpy64(float[] decoded_dhwc) throws IOException {
        File dir = new File(
                Environment.getExternalStoragePublicDirectory(Environment.DIRECTORY_DOWNLOADS),
                "DecoderAppResults/DecodedNPY64"
        );
        if (!dir.exists()) dir.mkdirs();

        File npy = new File(dir, "decoder64_" + System.currentTimeMillis() + ".npy");

        NpyWriter.writeNpy(npy.getAbsolutePath(), decoded_dhwc, new int[]{64, 64, 64, 4});

        Log.i("DecoderApp", "💾 Saved 64-ch NPY: " + npy.getAbsolutePath());
        return npy;
    }
}
