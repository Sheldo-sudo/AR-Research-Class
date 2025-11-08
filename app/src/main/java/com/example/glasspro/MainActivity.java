package com.example.glasspro;

import static com.example.glasspro.NativeProcessor.dehaze;
import static com.example.glasspro.NativeProcessor.enhance;
import static com.example.glasspro.NativeProcessor.enhanceByCLAHE;
import static com.example.glasspro.NativeProcessor.enhanceByMSRCR;
import static com.example.glasspro.NativeProcessor.videoStab;
import static com.example.glasspro.NativeProcessor.detectMotion;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import androidx.core.app.ActivityCompat;

import org.jetbrains.annotations.NotNull;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    // --- [!! 关键：静态代码块只负责加载库，不创建 MatOfRect !!] ---
    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "STATIC: OpenCV library load failed!");
        } else {
            Log.d(TAG, "STATIC: OpenCV library loaded successfully!");
        }
        try {
            System.loadLibrary("dehaze");
            System.loadLibrary("enhance");
            System.loadLibrary("threshold");
            System.loadLibrary("stab");
            System.loadLibrary("motion");
            System.loadLibrary("native-lib");
            Log.d(TAG, "STATIC: All native libraries loaded successfully");
        } catch (UnsatisfiedLinkError e) {
            Log.e(TAG, "STATIC: Failed to load native libraries: " + e.getMessage());
        }
    }
    // --- [!! 静态代码块结束 !!] ---


    private static final int CAMERA_PERMISSION_REQUEST = 1;
    private static final int MAX_FRAME_WIDTH = 1280;
    private static final int MAX_FRAME_HEIGHT = 720;
    private static final double NOISE_LEVEL = 10.0;

    public enum ProcessingMode {
        NONE,
        ENHANCE,
        DEHAZE,
        CLAHE,
        MSRCR
    }
    // 状态变量
    private ProcessingMode currentMode = ProcessingMode.NONE;
    private boolean isFirstFrame = true;
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat lastFrameForStab;
    private Mat lastFrameForMotion;
    private Mat lastFrameForEnhance;
    // UI组件
    private Button btnEnhance;
    private Button btnDehaze;
    private Button btnCLAHE;
    private Button btnMSRCR;

    // --- [!! 运动检测变量 !!] ---
    private boolean isMotionEnabled = true;
    private int motionFrameCounter = 0;
    private static final int MOTION_DETECTION_STRIDE = 6;

    // --- [!! 核心修改：这里不再直接 new MatOfRect() !!] ---
    private MatOfRect lastDetectedBoxes = null; // 现在它被初始化为 null
    private Rect[] lastBoxesArray = new Rect[0];
    // --- [!! 修改结束 !!] ---


    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        if (!initializeOpenCV()) { // 这里的 OpenCVLoader.initDebug() 也是一个很好的运行时检查
            return;
        }

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        requestPermissions();
        initializeUI();
    }

    private boolean initializeOpenCV() {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV (C++) library initialization failed! (Static block FAILED)");
            Toast.makeText(this, "OpenCV (C++) library initialization failed!", Toast.LENGTH_LONG).show();
            return false;
        }
        Log.i(TAG, "OpenCV (C++) library loaded successfully (from static block check)");
        return true;
    }

    private void requestPermissions() {
        ActivityCompat.requestPermissions(
                this,
                new String[]{Manifest.permission.CAMERA},
                CAMERA_PERMISSION_REQUEST
        );
    }
    private void initializeUI() {
        setContentView(R.layout.activity_main);
        initializeCamera();
        initializeButtons();
        setRequestedOrientation(ActivityInfo.SCREEN_ORIENTATION_LANDSCAPE);
    }
    private void initializeCamera() {
        mOpenCvCameraView = findViewById(R.id.main_surface);
        mOpenCvCameraView.setVisibility(SurfaceView.VISIBLE);
        mOpenCvCameraView.setCvCameraViewListener(this);
        mOpenCvCameraView.setCameraIndex(0);
        mOpenCvCameraView.setMaxFrameSize(MAX_FRAME_WIDTH, MAX_FRAME_HEIGHT);
    }
    @SuppressLint("SetTextI18n")
    private void initializeButtons() {
        btnEnhance = findViewById(R.id.button1);
        btnDehaze = findViewById(R.id.button2);
        btnCLAHE = findViewById(R.id.button3);
        btnMSRCR = findViewById(R.id.button4);

        setupButton(btnEnhance, "Enhance", ProcessingMode.ENHANCE);
        setupButton(btnDehaze, "Dehaze", ProcessingMode.DEHAZE);
        setupButton(btnCLAHE, "CLAHE", ProcessingMode.CLAHE);
        setupButton(btnMSRCR, "MSRCR", ProcessingMode.MSRCR);
    }
    private void setupButton(Button button, String text, ProcessingMode mode) {
        button.setText(text);
        button.setOnClickListener(v -> {
            if (currentMode != mode) {
                isFirstFrame = true;
            }
            currentMode = (currentMode == mode) ? ProcessingMode.NONE : mode;
            updateButtonStates();
        });
    }
    private void updateButtonStates() {
        btnEnhance.setSelected(currentMode == ProcessingMode.ENHANCE);
        btnDehaze.setSelected(currentMode == ProcessingMode.DEHAZE);
        btnCLAHE.setSelected(currentMode == ProcessingMode.CLAHE);
        btnMSRCR.setSelected(currentMode == ProcessingMode.MSRCR);
    }
    @Override
    public void onRequestPermissionsResult(int requestCode, @NotNull String[] permissions,
                                           @NotNull int[] grantResults) {
        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                mOpenCvCameraView.setCameraPermissionGranted();
            } else {
                String message = "Camera permission was not granted";
                Log.e(TAG, message);
//                Toast.makeText(this, message, Toast.LENGTH_LONG).show();
            }
        } else {
            Log.e(TAG, "Unexpected permission request");
        }
    }
    // ==================== 图像处理管道 ====================
    @Override
    public Mat onCameraFrame(CvCameraViewFrame frame) {
        Mat inputFrame = frame.rgba();
        try {
            // --- [!! 核心修改：懒加载 MatOfRect !!] ---
            if (lastDetectedBoxes == null) {
                // 只有当 C++ 库确定加载成功后，才在这里首次创建 MatOfRect
                // 它会在 onCameraFrame 第一次运行时被创建
                lastDetectedBoxes = new MatOfRect();
                Log.d(TAG, "lastDetectedBoxes created lazily on first frame.");
            }
            // --- [!! 修改结束 !!] ---

            // 1. 稳像处理 (每帧运行)
            processStab(inputFrame);

            // 2. 按键增强处理 (每帧运行)
            processFrame(inputFrame);

            // 3. 自动的、节流的运动 "检测" (昂贵)
            processMotion_Detection(inputFrame);

            // 4. 自动的、每帧的运动 "绘制" (廉价)
            drawMotion_Drawing(inputFrame);

        } catch (Exception e) {
            Log.e(TAG, "Error processing frame: " + e.getMessage());
        }
        return inputFrame;
    }
    private void processFrame(Mat frame) {
        switch (currentMode) {
            case ENHANCE:
                processEnhance(frame);
                break;
            case DEHAZE:
                processDehaze(frame);
                break;
            case CLAHE:
                processCLAHE(frame);
                break;
            case MSRCR:
                processMSRCR(frame);
                break;
            default:
                break;
        }
    }

    /**
     * 在后台自动运行的、节流的运动 "检测" (昂贵)
     */
    private void processMotion_Detection(Mat frame) {
        if (isMotionEnabled && lastDetectedBoxes != null) { // [!! 额外检查 !!]
            motionFrameCounter++;
            if (motionFrameCounter % MOTION_DETECTION_STRIDE == 0) {
                if (motionFrameCounter > 1000) motionFrameCounter = 0;

                Mat old = new Mat();
                frame.copyTo(old);

                if (lastFrameForMotion != null && !lastFrameForMotion.empty()) {
                    detectMotion(
                            lastFrameForMotion.getNativeObjAddr(),
                            frame.getNativeObjAddr(),
                            lastDetectedBoxes.getNativeObjAddr()
                    );
                    lastBoxesArray = lastDetectedBoxes.toArray();
                }

                if (lastFrameForMotion == null) {
                    lastFrameForMotion = new Mat();
                }
                old.copyTo(lastFrameForMotion);
                old.release();
            }
        }
    }

    /**
     * 在 "每帧" 运行的运动 "绘制" (廉价)
     */
    private void drawMotion_Drawing(Mat frame) {
        if (isMotionEnabled && lastBoxesArray != null && lastBoxesArray.length > 0) {
            for (Rect box : lastBoxesArray) {
                Imgproc.rectangle(frame, box, new Scalar(0, 255, 0, 255), 3);
            }
        }
    }


    private void processEnhance(Mat frame) {
        Mat old = new Mat();
        frame.copyTo(old);
        if (!isFirstFrame && lastFrameForEnhance != null && !lastFrameForEnhance.empty()) {
            enhance(lastFrameForEnhance.getNativeObjAddr(), frame.getNativeObjAddr(), NOISE_LEVEL);
        }
        isFirstFrame = false;
        if (lastFrameForEnhance == null) {
            lastFrameForEnhance = new Mat();
        }
        old.copyTo(lastFrameForEnhance);

    }
    private void processDehaze(Mat frame) {
        if (frame != null && !frame.empty()) {
            dehaze(frame.getNativeObjAddr());
        }
    }
    private void processCLAHE(Mat frame) {
        if (frame != null && !frame.empty()) {
            enhanceByCLAHE(frame.getNativeObjAddr());
        }
    }
    private void processMSRCR(Mat frame) {
        if (frame != null && !frame.empty()) {
            enhanceByMSRCR(frame.getNativeObjAddr());
        }
    }
    private void processStab(Mat frame) {
        Mat old = new Mat();
        frame.copyTo(old);

        if (!isFirstFrame && lastFrameForStab != null && !lastFrameForStab.empty()) {
            videoStab(lastFrameForStab.getNativeObjAddr(), frame.getNativeObjAddr());
        }
        isFirstFrame = false;
        if (lastFrameForStab == null) {
            lastFrameForStab = new Mat();
        }
        old.copyTo(lastFrameForStab);

    }
    //在相机启动时
    @Override
    public void onCameraViewStarted(int width, int height) {
        // [!! 修改：这里不再重新创建 lastDetectedBoxes !!]
        // 它会通过 onCameraFrame 中的懒加载模式自动创建。
        isFirstFrame = true;
        if (lastFrameForStab != null) { lastFrameForStab.release(); lastFrameForStab = null; }
        if (lastFrameForMotion != null) { lastFrameForMotion.release(); lastFrameForMotion = null; }
        if (lastFrameForEnhance != null) { lastFrameForEnhance.release(); lastFrameForEnhance = null; }
        lastBoxesArray = new Rect[0]; // 清空数组
    }

    @Override
    public void onCameraViewStopped() {
        releaseResources();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }

    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.enableView();
        }
        // [!! 修改：不再在这里重置 isFirstFrame, 因为 onCameraViewStarted 会做 !!]
        // isFirstFrame = true;
        currentMode = ProcessingMode.NONE;
        updateButtonStates();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
        releaseResources();
    }

    private void releaseResources() {
        try {
            if (lastFrameForStab != null) {
                lastFrameForStab.release();
                lastFrameForStab = null;
            }
            if (lastFrameForEnhance != null) {
                lastFrameForEnhance.release();
                lastFrameForEnhance = null;
            }
            if (lastFrameForMotion != null) {
                lastFrameForMotion.release();
                lastFrameForMotion = null;
            }
            // --- [!! 仍然释放 !!] ---
            if (lastDetectedBoxes != null) {
                lastDetectedBoxes.release();
                lastDetectedBoxes = null;
            }
            // --- [!! 释放结束 !!] ---

            lastBoxesArray = new Rect[0]; // 清空数组

        } catch (Exception e) {
            Log.e(TAG, "Error releasing resources: " + e.getMessage());
        }
    }
}