package com.example.decoderapp;

import java.io.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

public class NpyWriter {

    /** 标准 NPY Header 生成 */
    private static byte[] makeHeader(int[] shape) throws IOException {
        StringBuilder shp = new StringBuilder("(");
        for (int i = 0; i < shape.length; i++) {
            shp.append(shape[i]);
            if (shape.length == 1) shp.append(",");
            if (i < shape.length - 1) shp.append(", ");
        }
        shp.append(")");

        String dict = "{'descr': '<f4', 'fortran_order': False, 'shape': " + shp + ", }";
        int pad = 16 - ((10 + dict.length()) % 16);
        if (pad == 16) pad = 0;
        /* 2025.11.11 Update: header 没有以 \n 结束
           NumPy 文件头必须这样结尾：
           {'descr': '<f4', 'fortran_order': False, 'shape': (128, 128, 128, 4), }\n
            也就是说：最后一个字符必须是 换行符 \n；
            header 总长度（包括换行）必须是 16 字节对齐。
            缺少 \n，虽然总字节数对齐，但 header 实际结束位置晚了 1 个字节。
            导致数据区起始位置 错位 1 字节。
            Android 在按 float 解析时跳过了头一个字节，整份数据偏移 → 尾部多出 3 个无效字节。
            */
        String header = dict + " ".repeat(pad) + "\n";

        ByteArrayOutputStream baos = new ByteArrayOutputStream();
        baos.write(0x93);
        baos.write("NUMPY".getBytes("ASCII"));
        baos.write(1); baos.write(0); // version 1.0
        baos.write(header.length() & 0xFF);
        baos.write((header.length() >> 8) & 0xFF);
        baos.write(header.getBytes("ASCII"));
        return baos.toByteArray();
    }

    public static void writeNpy(String path, float[] data, int[] shape) throws IOException {
        try (FileOutputStream fos = new FileOutputStream(path)) {
            fos.write(makeHeader(shape));
            ByteBuffer bb = ByteBuffer.allocate(data.length * 4).order(ByteOrder.LITTLE_ENDIAN);
            bb.asFloatBuffer().put(data);
            fos.write(bb.array());
        }
    }

}
