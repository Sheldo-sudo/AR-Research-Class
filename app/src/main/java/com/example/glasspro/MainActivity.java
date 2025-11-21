package com.example.glasspro;

import static com.example.glasspro.NativeProcessor.*; // 静态导入所有功能

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.graphics.Bitmap;
import android.graphics.Color;
import android.graphics.PorterDuff;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.android.Utils;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    static {
        if (!OpenCVLoader.initDebug()) Log.e(TAG, "OpenCV load failed!");
        try {
            System.loadLibrary("dehaze");
            System.loadLibrary("enhance");
            System.loadLibrary("threshold");
            System.loadLibrary("stab");
            System.loadLibrary("motion");
            System.loadLibrary("native-lib");
            System.loadLibrary("vision_processor");
        } catch (UnsatisfiedLinkError e) { Log.e(TAG, "Libs load failed"); }
    }

    private static final int CAMERA_PERMISSION_REQUEST = 1;
    private static final double NOISE_LEVEL = 10.0;

    // 增强模式
    public enum ImageEnhanceMode { NONE, ENHANCE, DEHAZE, CLAHE, MSRCR }

    private ImageEnhanceMode currentEnhanceMode = ImageEnhanceMode.NONE;
    private boolean isDetectionEnabled = false;
    private boolean isTakePhoto = false;
    private boolean isFirstFrame = true;

    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat lastFrameForStab, lastFrameForEnhance;

    // 缓存对象 (为了避免GC，依然在Activity持有，或者移入Processor也可，这里保留在Activity比较稳妥)
    private MatOfRect detectionRectBuffer;
    private MatOfInt detectionIdBuffer;

    // DNN
    private long dnnNetPtr = 0;
    private static final String MODEL_PROTO = "mobilenet_ssd.prototxt";
    private static final String MODEL_WEIGHTS = "mobilenet_ssd.caffemodel";

    // UI Buttons
    private Button btnEnhance, btnDehaze, btnCLAHE, btnMSRCR, btnDetect, btnPhoto;

    // Colors (UI only)
    private final int COLOR_ACTIVE_BG = 0xFF00B0FF;
    private final int COLOR_INACTIVE_BG = 0xFF37474F;
    private final int COLOR_PHOTO_BG = 0xFFFF6E40;
    private final int COLOR_TEXT_ACTIVE = Color.WHITE;
    private final int COLOR_TEXT_INACTIVE = 0xFFB0BEC5;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        setContentView(R.layout.activity_main);
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);

        initializeCamera();
        initializeButtons();
        requestPermissions();
    }

    private void initializeCamera() {
        mOpenCvCameraView = findViewById(R.id.main_surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(0);
        mOpenCvCameraView.setMaxFrameSize(1280, 720);
    }

    private void requestPermissions() {
        ActivityCompat.requestPermissions(this, new String[]{Manifest.permission.CAMERA}, CAMERA_PERMISSION_REQUEST);
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, String[] permissions, int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST && grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
            mOpenCvCameraView.setCameraPermissionGranted();
        }
    }

    @SuppressLint("SetTextI1n")
    private void initializeButtons() {
        btnEnhance = findViewById(R.id.button1);
        btnDehaze = findViewById(R.id.button2);
        btnCLAHE = findViewById(R.id.button3);
        btnMSRCR = findViewById(R.id.button4);
        btnDetect = findViewById(R.id.button_detect);
        btnPhoto = findViewById(R.id.button_photo);

        setupEnhanceButton(btnEnhance, ImageEnhanceMode.ENHANCE);
        setupEnhanceButton(btnDehaze, ImageEnhanceMode.DEHAZE);
        setupEnhanceButton(btnCLAHE, ImageEnhanceMode.CLAHE);
        setupEnhanceButton(btnMSRCR, ImageEnhanceMode.MSRCR);

        btnDetect.setOnClickListener(v -> {
            isDetectionEnabled = !isDetectionEnabled;
            if (!isDetectionEnabled) {
                // 关键：关闭时清理 NativeProcessor 里的状态
                NativeProcessor.resetStabilizer();
            }
            updateButtonStyles();
        });

        btnPhoto.setOnClickListener(v -> {
            isTakePhoto = true;
            Toast.makeText(this, "正在拍照...", Toast.LENGTH_SHORT).show();
        });
        updateButtonStyles();
    }

    private void setupEnhanceButton(Button button, ImageEnhanceMode mode) {
        button.setOnClickListener(v -> {
            currentEnhanceMode = (currentEnhanceMode == mode) ? ImageEnhanceMode.NONE : mode;
            isFirstFrame = true; // 切换模式重置稳像参考帧
            updateButtonStyles();
        });
    }

    private void updateButtonStyles() {
        updateSingleButtonStyle(btnEnhance, currentEnhanceMode == ImageEnhanceMode.ENHANCE);
        updateSingleButtonStyle(btnDehaze, currentEnhanceMode == ImageEnhanceMode.DEHAZE);
        updateSingleButtonStyle(btnCLAHE, currentEnhanceMode == ImageEnhanceMode.CLAHE);
        updateSingleButtonStyle(btnMSRCR, currentEnhanceMode == ImageEnhanceMode.MSRCR);
        updateSingleButtonStyle(btnDetect, isDetectionEnabled);
        btnPhoto.getBackground().setColorFilter(COLOR_PHOTO_BG, PorterDuff.Mode.MULTIPLY);
        btnPhoto.setTextColor(Color.WHITE);
    }

    private void updateSingleButtonStyle(Button btn, boolean isActive) {
        int bgColor = isActive ? COLOR_ACTIVE_BG : COLOR_INACTIVE_BG;
        int txtColor = isActive ? COLOR_TEXT_ACTIVE : COLOR_TEXT_INACTIVE;
        btn.getBackground().setColorFilter(bgColor, PorterDuff.Mode.MULTIPLY);
        btn.setTextColor(txtColor);
    }

    // ==================== 核心循环 ====================
    @Override
    public Mat onCameraFrame(CvCameraViewFrame frame) {
        Mat inputFrame = frame.rgba();

        // 延迟初始化缓存
        if (detectionRectBuffer == null) detectionRectBuffer = new MatOfRect();
        if (detectionIdBuffer == null) detectionIdBuffer = new MatOfInt();

        try {
            // 1. 视频防抖 (NativeProcessor static call)
            processStab(inputFrame);

            // 2. 图像增强
            processEnhancementModes(inputFrame);

            // 3. 目标检测
            if (isDetectionEnabled && dnnNetPtr != 0) {
                // 只有一行代码：检测+防抖+绘制 全包了
                NativeProcessor.detectAndDraw(inputFrame, dnnNetPtr, detectionRectBuffer, detectionIdBuffer, 0.3f);
            }

            // 4. 拍照
            if (isTakePhoto) {
                isTakePhoto = false;
                saveImageToGallery(inputFrame);
            }
        } catch (Exception e) {
            Log.e(TAG, "Frame Error: " + e.getMessage());
        }
        return inputFrame;
    }

    // --- 辅助逻辑 ---
    private void processEnhancementModes(Mat frame) {
        switch (currentEnhanceMode) {
            case ENHANCE: processEnhanceWithState(frame); break;
            case DEHAZE: if(!frame.empty()) dehaze(frame.getNativeObjAddr()); break;
            case CLAHE: if(!frame.empty()) enhanceByCLAHE(frame.getNativeObjAddr()); break;
            case MSRCR: if(!frame.empty()) enhanceByMSRCR(frame.getNativeObjAddr()); break;
            default: break;
        }
    }

    private void processEnhanceWithState(Mat frame) {
        Mat old = new Mat(); frame.copyTo(old);
        if (!isFirstFrame && lastFrameForEnhance != null && !lastFrameForEnhance.empty()) {
            enhance(lastFrameForEnhance.getNativeObjAddr(), frame.getNativeObjAddr(), NOISE_LEVEL);
        }
        if (lastFrameForEnhance == null) lastFrameForEnhance = new Mat();
        old.copyTo(lastFrameForEnhance);
    }

    private void processStab(Mat frame) {
        Mat old = new Mat(); frame.copyTo(old);
        if (!isFirstFrame && lastFrameForStab != null && !lastFrameForStab.empty()) {
            videoStab(lastFrameForStab.getNativeObjAddr(), frame.getNativeObjAddr());
        }
        isFirstFrame = false;
        if (lastFrameForStab == null) lastFrameForStab = new Mat();
        old.copyTo(lastFrameForStab);
    }

    private void saveImageToGallery(Mat mat) {
        final Bitmap bitmap = Bitmap.createBitmap(mat.cols(), mat.rows(), Bitmap.Config.ARGB_8888);
        Utils.matToBitmap(mat, bitmap);
        new Thread(() -> {
            try {
                String fileName = "GlassPro_" + System.currentTimeMillis() + ".jpg";
                android.content.ContentValues values = new android.content.ContentValues();
                values.put(android.provider.MediaStore.Images.Media.DISPLAY_NAME, fileName);
                values.put(android.provider.MediaStore.Images.Media.MIME_TYPE, "image/jpeg");
                values.put(android.provider.MediaStore.Images.Media.RELATIVE_PATH, android.os.Environment.DIRECTORY_PICTURES + "/GlassPro");
                android.net.Uri uri = getContentResolver().insert(android.provider.MediaStore.Images.Media.EXTERNAL_CONTENT_URI, values);
                if (uri != null) {
                    try (OutputStream out = getContentResolver().openOutputStream(uri)) {
                        bitmap.compress(Bitmap.CompressFormat.JPEG, 95, out);
                    }
                    runOnUiThread(() -> Toast.makeText(MainActivity.this, "Saved to Gallery", Toast.LENGTH_SHORT).show());
                }
            } catch (Exception e) { Log.e(TAG, "Save error: " + e.getMessage()); }
        }).start();
    }

    @Override
    public void onCameraViewStarted(int width, int height) {
        isFirstFrame = true;
        if (dnnNetPtr == 0) {
            new Thread(() -> {
                try {
                    String proto = getPathFromAsset(MODEL_PROTO);
                    String model = getPathFromAsset(MODEL_WEIGHTS);
                    if (proto != null && model != null) dnnNetPtr = loadObjectDetector(proto, model);
                } catch (Exception e) { Log.e(TAG, "DNN Load Error"); }
            }).start();
        }
    }

    @Override
    public void onCameraViewStopped() {
        if (lastFrameForStab != null) lastFrameForStab.release();
        if (lastFrameForEnhance != null) lastFrameForEnhance.release();
        if (detectionRectBuffer != null) detectionRectBuffer.release();
        if (detectionIdBuffer != null) detectionIdBuffer.release();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
        if (dnnNetPtr != 0) releaseObjectDetector(dnnNetPtr);
    }

    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null) mOpenCvCameraView.enableView();
        currentEnhanceMode = ImageEnhanceMode.NONE;
        isDetectionEnabled = false;
        NativeProcessor.resetStabilizer(); // 切回来时重置状态
        updateButtonStyles();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) mOpenCvCameraView.disableView();
    }

    private String getPathFromAsset(String assetName) {
        File outFile = new File(getCacheDir(), assetName);
        if (outFile.exists() && outFile.length() > 0) return outFile.getAbsolutePath();
        try (InputStream in = getAssets().open(assetName); OutputStream out = new FileOutputStream(outFile)) {
            byte[] buffer = new byte[4096];
            int read;
            while ((read = in.read(buffer)) != -1) out.write(buffer, 0, read);
            return outFile.getAbsolutePath();
        } catch (Exception e) { return null; }
    }
}