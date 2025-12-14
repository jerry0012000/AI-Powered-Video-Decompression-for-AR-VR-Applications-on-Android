package com.example.decoderapp;

import android.content.Context;
import android.database.Cursor;
import android.net.Uri;
import android.provider.OpenableColumns;
import android.util.Log;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;

public class FileUtil {

    /** 从 SAF Uri 读全部字节（兼容 API 26） */
    public static byte[] readAll(Context ctx, Uri uri) throws IOException {
        InputStream is = ctx.getContentResolver().openInputStream(uri);

        ByteArrayOutputStream bos = new ByteArrayOutputStream();
        byte[] buf = new byte[8192];
        int n;
        while ((n = is.read(buf)) > 0)
            bos.write(buf, 0, n);

        is.close();
        return bos.toByteArray();
    }

    /** 读取 (1,4,4,4,128) 的 NPY 并返回 DHWC 扁平（8192 floats） */
    public static float[] loadEmbedNPY_DHWC(Context ctx, Uri uri) throws IOException {
        byte[] bytes = readAll(ctx, uri);

        // 极简 .npy 解析：略过 header，找到第一个 '[' 到 ']' 的 shape，然后找 data 起点。
        // 为简单可靠，这里假设是 NumPy 1.0 规范 + '<f4' 小端 float32 + 非 Fortran
        int magic = (bytes[0] & 0xFF);
        if (magic != 0x93) throw new IOException("Not an NPY file (magic mismatch)");

        int major = bytes[6] & 0xFF;
        int minor = bytes[7] & 0xFF;
        int headerLen;
        if (major == 1) {
            headerLen = ((bytes[9] & 0xFF) << 8) | (bytes[8] & 0xFF);
        } else if (major == 2) {
            headerLen = ((bytes[9] & 0xFF) << 8) | (bytes[8] & 0xFF); // 简化处理
        } else {
            headerLen = ((bytes[9] & 0xFF) << 8) | (bytes[8] & 0xFF);
        }
        int headerStart = 10;
        int dataStart = headerStart + headerLen;

        String header = new String(bytes, headerStart, headerLen, "ASCII");
        if (!header.contains("<f4")) throw new IOException("NPY not float32");
        if (!header.contains("fortran_order')") && !header.contains("fortran_order\":")) {
            // 放宽检查
        }

        int dataBytes = bytes.length - dataStart;
        int expectedFloats = 4 * 4 * 4 * 128; // = 8192
        int expectedBytes = expectedFloats * 4; // = 32768
        if (dataBytes != expectedBytes) {
            // 大部分 numpy 的 (1,4,4,4,128) 都是 8192 floats
            Log.w("DecoderApp", "NPY payload = "+dataBytes+" bytes (expect 32768)");
            throw new IOException(
                    "Latent file is not 4x4x4x128: payload = "
                            + dataBytes + " bytes, expected " + expectedBytes
                            + " (maybe this is 4x4x4x64 or other shape?)");
        }

        float[] out = new float[8192];
        ByteBuffer.wrap(bytes, dataStart, Math.min(dataBytes, 8192*4))
                .order(ByteOrder.LITTLE_ENDIAN).asFloatBuffer().get(out);
        return out; // 这是按 (D,H,W,C) 扁平（DHWC），需再转成 NCDHW
    }

    public static float[] loadNpyFloat(Context ctx, Uri uri) throws IOException {
        try (InputStream is = ctx.getContentResolver().openInputStream(uri);
             DataInputStream dis = new DataInputStream(new BufferedInputStream(is))) {

            // 跳过 header（动态计算）
            ByteArrayOutputStream headerBytes = new ByteArrayOutputStream();
            int b;
            while ((b = dis.read()) != -1) {
                headerBytes.write(b);
                // header 以 '\n' 结尾
                if (headerBytes.size() > 8 && headerBytes.toByteArray()[headerBytes.size() - 1] == '\n')
                    break;
            }

            // 剩余部分全部读为 float
            ByteArrayOutputStream data = new ByteArrayOutputStream();
            byte[] buf = new byte[8192];
            int len;
            while ((len = dis.read(buf)) != -1) {
                data.write(buf, 0, len);
            }

            byte[] arr = data.toByteArray();
            int n = arr.length / 4;
            float[] out = new float[n];
            ByteBuffer bb = ByteBuffer.wrap(arr).order(ByteOrder.LITTLE_ENDIAN);
            bb.asFloatBuffer().get(out);

            Log.i("FileUtil", "✅ Loaded " + n + " floats");
            return out;
        }
    }

    /** 读取 (1,4,4,4,64) 的 NPY 并返回 DHWC 扁平（4096 floats） */
    public static float[] loadEmbedNPY_DHWC_64(Context ctx, Uri uri) throws IOException {

        byte[] bytes = readAll(ctx, uri);

        // -------- Parse NPY header exactly the same way --------
        int magic = (bytes[0] & 0xFF);
        if (magic != 0x93) throw new IOException("Not an NPY file");

        int major = bytes[6] & 0xFF;
        int minor = bytes[7] & 0xFF;

        int headerLen;
        if (major == 1) {
            headerLen = ((bytes[9] & 0xFF) << 8) | (bytes[8] & 0xFF);
        } else {
            headerLen = ((bytes[9] & 0xFF) << 8) | (bytes[8] & 0xFF);
        }

        int headerStart = 10;
        int dataStart = headerStart + headerLen;

        String header = new String(bytes, headerStart, headerLen, "ASCII");

        if (!header.contains("<f4"))
            throw new IOException("NPY is not float32");

        // 目标 shape = (4,4,4,64)
        int expectedFloats = 4 * 4 * 4 * 64; // = 4096
        int expectedBytes = expectedFloats * 4; // = 16384

        int dataBytes = bytes.length - dataStart;

        // 目标：严格只接受 (4,4,4,64) = 4096 floats = 16384 bytes 的 latent
        if (dataBytes != expectedBytes) {
            Log.w("DecoderApp", "NPY payload = "+dataBytes+" bytes (expect 16384)");
            throw new IOException(
                    "Latent file is not 4x4x4x64: payload = "
                            + dataBytes + " bytes, expected " + expectedBytes
                            + " (maybe this is 4x4x4x128 or other shape?)");
        }

        // -------- Read float values --------
        float[] out = new float[expectedFloats];

        ByteBuffer.wrap(bytes, dataStart, expectedBytes)
                .order(ByteOrder.LITTLE_ENDIAN)
                .asFloatBuffer()
                .get(out);

        Log.i("FileUtil", "✅ Loaded latent64 (4x4x4x64): " + out.length + " floats");
        return out;

    }

    /* 2025.11.13 Update: Display file name on screen*/
    public static String getFileName(Context ctx, Uri uri) {
        String name = null;

        // 1. Try DISPLAY_NAME
        if ("content".equals(uri.getScheme())) {
            Cursor cursor = ctx.getContentResolver().query(uri, null, null, null, null);
            try {
                if (cursor != null && cursor.moveToFirst()) {
                    int index = cursor.getColumnIndex(OpenableColumns.DISPLAY_NAME);
                    if (index >= 0) {
                        name = cursor.getString(index);
                    }
                }
            } finally {
                if (cursor != null) cursor.close();
            }
        }

        if (name == null) {
            name = uri.getLastPathSegment(); // fallback
        }

        // ----------------------------------------------------------
        // 2. Extract parent folder from uri.getPath() (virtual path)
        // ----------------------------------------------------------
        String path = uri.getPath();  // e.g. "/document/primary:Download/DecoderAppResults/file.npy"
        if (path != null) {
            // Extract the rightmost meaningful path portion
            int idx = path.lastIndexOf(':');
            if (idx >= 0 && idx < path.length() - 1) {
                path = path.substring(idx + 1);
                // now path = "Download/DecoderAppResults/file.npy"
            }
        }

        // 3. Combine folder + name
        if (path != null && path.contains("/")) {
            int slash = path.lastIndexOf('/');
            if (slash > 0) {
                String parent = path.substring(0, slash);
                return parent + "/" + name;   // e.g. "Download/DecoderAppResults/file.npy"
            }
        }

        // Default
        return name;
    }

    /* 2025.12.4 Update: Add getDisplayPath() */
    /**
     * 加载latent codes (.pt) 文件
     */
    public static float[] loadLatentCodes(Context context, Uri uri) {
        try (InputStream is = context.getContentResolver().openInputStream(uri)) {
            if (is == null) {
                Log.e("FileUtil", "Cannot open latent codes file");
                return new float[0];
            }

            byte[] bytes = readAllBytes(is);

            // 简化的PyTorch .pt 文件解析
            // 注意：真实场景需要完整解析PyTorch序列化格式
            return parseSimpleFloatArray(bytes);

        } catch (Exception e) {
            Log.e("FileUtil", "Failed to load latent codes", e);
            return new float[0];
        }
    }

    /**
     * 简化的浮点数组解析（假设是原始float数组）
     */
    private static float[] parseSimpleFloatArray(byte[] bytes) {
        if (bytes.length % 4 != 0) {
            Log.w("FileUtil", "Byte array length not divisible by 4");
            return new float[0];
        }

        int floatCount = bytes.length / 4;
        float[] floats = new float[floatCount];

        ByteBuffer buffer = ByteBuffer.wrap(bytes);
        buffer.order(ByteOrder.LITTLE_ENDIAN);

        FloatBuffer floatBuffer = buffer.asFloatBuffer();
        floatBuffer.get(floats);

        return floats;
    }

    /**
     * 读取InputStream所有字节
     */
    private static byte[] readAllBytes(InputStream is) throws IOException {
        ByteArrayOutputStream buffer = new ByteArrayOutputStream();
        byte[] data = new byte[4096];
        int nRead;
        while ((nRead = is.read(data, 0, data.length)) != -1) {
            buffer.write(data, 0, nRead);
        }
        buffer.flush();
        return buffer.toByteArray();
    }

    /**
     * 获取latent code子集（用于插值）
     */
    public static float[] getLatentCodeSubset(float[] allCodes, int codeIndex, int latentDim) {
        if (allCodes == null || codeIndex * latentDim + latentDim > allCodes.length) {
            Log.w("FileUtil", "Invalid latent code index or dimension");
            return new float[latentDim]; // 返回零数组
        }

        float[] subset = new float[latentDim];
        System.arraycopy(allCodes, codeIndex * latentDim, subset, 0, latentDim);
        return subset;
    }

}
