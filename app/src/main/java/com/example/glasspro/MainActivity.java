package com.example.glasspro;

import static com.example.glasspro.NativeProcessor.dehaze;
import static com.example.glasspro.NativeProcessor.enhance;
import static com.example.glasspro.NativeProcessor.enhanceByCLAHE;
import static com.example.glasspro.NativeProcessor.enhanceByMSRCR;
import static com.example.glasspro.NativeProcessor.videoStab;

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

public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";
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

    private Mat lastFrameForEnhance;
    // UI组件
    private Button btnEnhance;
    private Button btnDehaze;
    private Button btnCLAHE;
    private Button btnMSRCR;

    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);
        // 初始化OpenCV
        if (!initializeOpenCV()) {
            return;
        }
        // 保持屏幕常亮
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        requestPermissions();
        // 初始化UI
        initializeUI();
    }
    private boolean initializeOpenCV() {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "OpenCV initialization failed!");
            Toast.makeText(this, "OpenCV initialization failed!", Toast.LENGTH_LONG).show();
            return false;
        }
        Log.i(TAG, "OpenCV loaded successfully");
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
            processStab(inputFrame);
            processFrame(inputFrame);
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
            // Call the MSRCR enhancement method
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

    }
    @Override
    public void onCameraViewStopped() {
        releaseResources();
    }
    /**
     * 在暂停时停止OpenCV视图并释放资源。
     */
    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
    }
    /**
     * 在恢复时重新启用OpenCV视图。
     */
    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.enableView();
        }
        isFirstFrame = true;
        currentMode = ProcessingMode.NONE;
        updateButtonStates();
    }
    /**
     * 在销毁时释放所有资源。
     */
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

        } catch (Exception e) {
            Log.e(TAG, "Error releasing lastFrame: " + e.getMessage());
        }
    }
}
