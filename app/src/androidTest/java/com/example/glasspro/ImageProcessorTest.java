package com.example.glasspro;

import static org.junit.Assert.assertNotNull;

import android.content.ContentResolver;
import android.content.ContentValues;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.BitmapFactory;
import android.net.Uri;
import android.os.Environment;
import android.provider.MediaStore;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;

import org.junit.Test;
import org.junit.runner.RunWith;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;

import java.io.IOException;
import java.io.OutputStream;


@RunWith(AndroidJUnit4.class)
public class ImageProcessorTest {
    private static final int IMAGE=R.drawable.test6;//需要测试的图片

    @Test
    public void testEnhance() {
        Mat input = loadTestImage();
        Mat output = new Mat();
        assert input != null;
        input.copyTo(output);
        try {
            NativeProcessor.enhance(output.getNativeObjAddr(),input.getNativeObjAddr(),10.0);
        } catch (Exception e) {
            System.out.println("Enhancement failed: " + e.getMessage());
            return;
        }
        saveMatAsImage(output, "enhance_output.png");
    }
    @Test
    public void testDehaze() {
        Mat input = loadTestImage();
        assert input != null;
        NativeProcessor.dehaze(input.getNativeObjAddr());
        saveMatAsImage(input, "dehaze_output.png");
    }
    @Test
    public void testCLAHE() {
        Mat input = loadTestImage();
        assert input != null;
        NativeProcessor.enhanceByCLAHE(input.getNativeObjAddr());
        saveMatAsImage(input, "clahe_output.png");
    }
    @Test
    public void testMSRCR() {
        Mat input = loadTestImage();
        assert input != null;
        NativeProcessor.enhanceByMSRCR(input.getNativeObjAddr());
        saveMatAsImage(input, "msrcr_output.png");
    }
    private boolean initializeOpenCV() {
        return OpenCVLoader.initDebug();
    }
    // 通过 assets 目录加载测试图像
    private Mat loadTestImage() {
        if (!initializeOpenCV()) {
            return null;
        }
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        Bitmap bitmap = BitmapFactory.decodeResource(context.getResources(),IMAGE);
        if (bitmap == null) {
            throw new RuntimeException("Test image R.drawable.test not found or failed to decode.");
        }
        Mat mat = new Mat();
        Utils.bitmapToMat(bitmap, mat);
        return mat;
    }
    // 将 Mat 保存为 PNG 格式
    private void saveMatAsImage(Mat mat, String fileName) {
        Context context = InstrumentationRegistry.getInstrumentation().getTargetContext();
        // Step 1: Mat → Bitmap
        Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        // Step 2: 创建 ContentValues
        ContentValues values = new ContentValues();
        values.put(MediaStore.Images.Media.DISPLAY_NAME, fileName);
        values.put(MediaStore.Images.Media.MIME_TYPE, "image/png");
        values.put(MediaStore.Images.Media.RELATIVE_PATH, Environment.DIRECTORY_PICTURES + "/GlassProTest");
        // Step 3: 获取 URI 并写入图片内容
        ContentResolver resolver = context.getContentResolver();
        Uri uri = resolver.insert(MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);

        if (uri == null) {
            Log.e("MediaStore", "Failed to create new MediaStore record.");
            return;
        }

        try (OutputStream out = resolver.openOutputStream(uri)) {
            if (out != null) {
                bitmap.compress(Bitmap.CompressFormat.PNG, 100, out);
                Log.d("MediaStore", "Saved to gallery: " + fileName);
            } else {
                Log.e("MediaStore", "Failed to open output stream.");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

}
