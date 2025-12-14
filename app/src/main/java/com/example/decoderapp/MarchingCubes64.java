package com.example.decoderapp;

import android.util.Log;

import java.io.File;
import java.io.FileWriter;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

public class MarchingCubes64 {

    private final int D = 64, H = 64, W = 64;
    private final float[] sdf;        // D*H*W
    private final float[][] offsets;  // D*H*W x 3
    private final float iso;

    public MarchingCubes64(float[] sdf, float[][] offsets, float isoLevel) {
        this.sdf = sdf;
        this.offsets = offsets;
        this.iso = isoLevel;
    }

    private int idx(int z, int y, int x) {
        return z * H * W + y * W + x;
    }

    /** ------------------------
     * Main marching cubes
     * ------------------------ */
    public Result generate() {
        List<Float> verts = new ArrayList<>();
        List<Integer> faces = new ArrayList<>();

        for (int z = 0; z < D - 1; z++) {
            for (int y = 0; y < H - 1; y++) {
                for (int x = 0; x < W - 1; x++) {

                    int[] cornerIndex = new int[]{
                            idx(z, y, x),
                            idx(z, y, x + 1),
                            idx(z, y + 1, x + 1),
                            idx(z, y + 1, x),
                            idx(z + 1, y, x),
                            idx(z + 1, y, x + 1),
                            idx(z + 1, y + 1, x + 1),
                            idx(z + 1, y + 1, x)
                    };

                    float[] val = new float[8];
                    for (int i = 0; i < 8; i++) val[i] = sdf[cornerIndex[i]];

                    int cubeIndex = 0;
                    if (val[0] < iso) cubeIndex |= 1;
                    if (val[1] < iso) cubeIndex |= 2;
                    if (val[2] < iso) cubeIndex |= 4;
                    if (val[3] < iso) cubeIndex |= 8;
                    if (val[4] < iso) cubeIndex |= 16;
                    if (val[5] < iso) cubeIndex |= 32;
                    if (val[6] < iso) cubeIndex |= 64;
                    if (val[7] < iso) cubeIndex |= 128;

                    int edgeMask = edgeTable[cubeIndex];
                    if (edgeMask == 0) continue;

                    float[][] vertList = new float[12][3];

                    if ((edgeMask & 1) != 0) vertList[0] = vertInterp(x, y, z, x + 1, y, z, val[0], val[1], cornerIndex[0], cornerIndex[1]);
                    if ((edgeMask & 2) != 0) vertList[1] = vertInterp(x + 1, y, z, x + 1, y + 1, z, val[1], val[2], cornerIndex[1], cornerIndex[2]);
                    if ((edgeMask & 4) != 0) vertList[2] = vertInterp(x + 1, y + 1, z, x, y + 1, z, val[2], val[3], cornerIndex[2], cornerIndex[3]);
                    if ((edgeMask & 8) != 0) vertList[3] = vertInterp(x, y + 1, z, x, y, z, val[3], val[0], cornerIndex[3], cornerIndex[0]);
                    if ((edgeMask & 16) != 0) vertList[4] = vertInterp(x, y, z + 1, x + 1, y, z + 1, val[4], val[5], cornerIndex[4], cornerIndex[5]);
                    if ((edgeMask & 32) != 0) vertList[5] = vertInterp(x + 1, y, z + 1, x + 1, y + 1, z + 1, val[5], val[6], cornerIndex[5], cornerIndex[6]);
                    if ((edgeMask & 64) != 0) vertList[6] = vertInterp(x + 1, y + 1, z + 1, x, y + 1, z + 1, val[6], val[7], cornerIndex[6], cornerIndex[7]);
                    if ((edgeMask & 128) != 0) vertList[7] = vertInterp(x, y + 1, z + 1, x, y, z + 1, val[7], val[4], cornerIndex[7], cornerIndex[4]);
                    if ((edgeMask & 256) != 0) vertList[8] = vertInterp(x, y, z, x, y, z + 1, val[0], val[4], cornerIndex[0], cornerIndex[4]);
                    if ((edgeMask & 512) != 0) vertList[9] = vertInterp(x + 1, y, z, x + 1, y, z + 1, val[1], val[5], cornerIndex[1], cornerIndex[5]);
                    if ((edgeMask & 1024) != 0) vertList[10] = vertInterp(x + 1, y + 1, z, x + 1, y + 1, z + 1, val[2], val[6], cornerIndex[2], cornerIndex[6]);
                    if ((edgeMask & 2048) != 0) vertList[11] = vertInterp(x, y + 1, z, x, y + 1, z + 1, val[3], val[7], cornerIndex[3], cornerIndex[7]);

                    int[] tri = triTable[cubeIndex];
                    for (int t = 0; t < tri.length && tri[t] != -1; t += 3) {

                        int ia = addVertex(verts, vertList[tri[t]]);
                        int ib = addVertex(verts, vertList[tri[t + 1]]);
                        int ic = addVertex(verts, vertList[tri[t + 2]]);

                        faces.add(ia);
                        faces.add(ib);
                        faces.add(ic);
                    }
                }
            }
        }

        float[] vOut = new float[verts.size()];
        for (int i = 0; i < verts.size(); i++) vOut[i] = verts.get(i);

        int[] fOut = new int[faces.size()];
        for (int i = 0; i < faces.size(); i++) fOut[i] = faces.get(i);

        return new Result(vOut, fOut);
    }


    private int addVertex(List<Float> verts, float[] p) {
        int id = verts.size() / 3;
        verts.add(p[0]);
        verts.add(p[1]);
        verts.add(p[2]);
        return id;
    }

    private float[] vertInterp(int x1, int y1, int z1,
                               int x2, int y2, int z2,
                               float v1, float v2,
                               int idx1, int idx2) {

        float t = (iso - v1) / (v2 - v1 + 1e-12f);
        float px = x1 + t * (x2 - x1);
        float py = y1 + t * (y2 - y1);
        float pz = z1 + t * (z2 - z1);

        // ✨ 64³ offset（这里要开）✨
        float[] o1 = offsets[idx1];
        float[] o2 = offsets[idx2];
        px += (1 - t) * o1[0] + t * o2[0];
        py += (1 - t) * o1[1] + t * o2[1];
        pz += (1 - t) * o1[2] + t * o2[2];

        return new float[]{px, py, pz};
    }


    /** ----------------------------
     * Convert decoder output → MC64
     * ---------------------------- */
    public static MarchingCubes64 fromDecoderOutput64(float[] raw) {
        final int C = 4;
        int N = 64 * 64 * 64;
        int expected = N * C;        // 64³×4 = 1,048,576 floats

        if (raw.length < expected) {
            Log.e("MC64", "❌ raw too small: " + raw.length + " < " + expected);
            raw = Arrays.copyOf(raw, expected);
        } else if (raw.length > expected) {
            Log.w("MC64", "⚠ raw too big: " + raw.length + ", trimming to " + expected);
            raw = Arrays.copyOf(raw, expected);
        }

        float[] sdf = new float[N];
        float[][] offsets = new float[N][3];

        for (int i = 0; i < N; i++) {
            sdf[i] = raw[i * C];
            offsets[i][0] = raw[i * C + 1];
            offsets[i][1] = raw[i * C + 2];
            offsets[i][2] = raw[i * C + 3];
        }

        return new MarchingCubes64(sdf, offsets, 0.0f);
    }


    /** ------------------------
     * Save OBJ inside this class
     * ------------------------ */
    public static File saveObj(Result res) throws Exception {
        File dir = new File(
                android.os.Environment.getExternalStoragePublicDirectory(
                        android.os.Environment.DIRECTORY_DOWNLOADS),
                "DecoderAppResults/OBJ_64"
        );
        dir.mkdirs();

        File out = new File(dir, "mesh64_" + System.currentTimeMillis() + ".obj");
        FileWriter fw = new FileWriter(out);

        float[] v = res.vertices;
        int[] f = res.faces;

        for (int i = 0; i < v.length; i += 3) {
            fw.write("v " + v[i] + " " + v[i + 1] + " " + v[i + 2] + "\n");
        }

        for (int i = 0; i < f.length; i += 3) {
            fw.write("f " + (f[i] + 1) + " " + (f[i + 1] + 1) + " " + (f[i + 2] + 1) + "\n");
        }

        fw.close();
        return out;
    }


    /** Output struct */
    public static class Result {
        public final float[] vertices;
        public final int[] faces;

        public Result(float[] v, int[] f) {
            vertices = v;
            faces = f;
        }
    }


    /* ===================================================
     * MC lookup tables
     * =================================================== */
    private static final int[] edgeTable = MarchingCubes.edgeTable;
    private static final int[][] triTable = MarchingCubes.triTable;

}
