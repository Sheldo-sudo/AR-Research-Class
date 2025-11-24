package com.example.glasspro;

import static com.example.glasspro.NativeProcessor.dehaze;
import static com.example.glasspro.NativeProcessor.enhance;
import static com.example.glasspro.NativeProcessor.enhanceByCLAHE;
import static com.example.glasspro.NativeProcessor.enhanceByMSRCR;
import static com.example.glasspro.NativeProcessor.videoStab;

import android.Manifest;
import android.annotation.SuppressLint;
import android.content.pm.ActivityInfo;
import android.content.pm.PackageManager;
import android.hardware.Sensor;
import android.hardware.SensorEvent;
import android.hardware.SensorEventListener;
import android.hardware.SensorManager;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceView;
import android.view.WindowManager;
import android.widget.Button;
import android.widget.Toast;

import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;

import org.jetbrains.annotations.NotNull;
import org.opencv.android.CameraBridgeViewBase;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewFrame;
import org.opencv.android.CameraBridgeViewBase.CvCameraViewListener2;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";
    private static final int CAMERA_PERMISSION_REQUEST = 1;
    private static final int MAX_FRAME_WIDTH = 1280;
    private static final int MAX_FRAME_HEIGHT = 720;
    private static final double NOISE_LEVEL = 10.0;
    private static final float DETECTION_CONFIDENCE = 0.5f; // 检测置信度阈值

    // ==================== 处理模式枚举 ====================
    public enum ProcessingMode {
        NONE,
        ENHANCE,
        DEHAZE,
        CLAHE,
        MSRCR,
        MOTION_DETECT,    // 新增：运动检测
        OBJECT_DETECT     // 新增：目标检测
    }

    // ==================== 状态变量 ====================
    private ProcessingMode currentMode = ProcessingMode.NONE;
    private boolean isFirstFrame = true;
    private boolean isAutoModeEnabled = false;

    // OpenCV相关
    private CameraBridgeViewBase mOpenCvCameraView;
    private Mat lastFrameForStab;
    private Mat lastFrameForEnhance;
    private Mat lastFrameForMotion;  // 新增：运动检测用的上一帧

    // 传感器相关
    private SensorManager sensorManager;
    private Sensor accelerometer;
    private float accelX, accelY, accelZ;

    // 新增：目标检测相关
    private long detectorNetPtr = 0;
    private String modelProtoPath;
    private String modelWeightsPath;
    private MatOfRect detectedBoxes;      // 存储检测到的边界框
    private MatOfInt detectedClassIds;    // 存储检测到的类别ID

    // UI组件
    private Button btnEnhance;
    private Button btnDehaze;
    private Button btnCLAHE;
    private Button btnMSRCR;
    private Button btnAutoMode;
    private Button btnMotionDetect;   // 新增：运动检测按钮
    private Button btnObjectDetect;   // 新增：目标检测按钮

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

        // 初始化传感器
        initializeSensors();

        // 初始化检测器
        initializeDetector();
    }

    // ==================== 初始化方法 ====================

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

    /**
     * 初始化传感器
     */
    private void initializeSensors() {
        sensorManager = (SensorManager) getSystemService(SENSOR_SERVICE);
        accelerometer = sensorManager.getDefaultSensor(Sensor.TYPE_ACCELEROMETER);

        if (accelerometer != null) {
            sensorManager.registerListener(sensorEventListener, accelerometer,
                    SensorManager.SENSOR_DELAY_NORMAL);
            Log.i(TAG, "Accelerometer initialized successfully");
        } else {
            Log.w(TAG, "Accelerometer not available");
        }
    }

    /**
     * 传感器事件监听器
     */
    private final SensorEventListener sensorEventListener = new SensorEventListener() {
        @Override
        public void onSensorChanged(SensorEvent event) {
            if (event.sensor.getType() == Sensor.TYPE_ACCELEROMETER) {
                accelX = event.values[0];
                accelY = event.values[1];
                accelZ = event.values[2];
            }
        }

        @Override
        public void onAccuracyChanged(Sensor sensor, int accuracy) {
            // 不需要处理
        }
    };

    /**
     * 初始化目标检测器
     */
    private void initializeDetector() {
        try {
            // 设置模型文件路径（需要将模型文件放到assets目录）
            modelProtoPath = copyAssetToCache("MobileNetSSD_deploy.prototxt");
            modelWeightsPath = copyAssetToCache("MobileNetSSD_deploy.caffemodel");

            // 加载模型
            detectorNetPtr = NativeProcessor.loadObjectDetector(modelProtoPath, modelWeightsPath);

            if (detectorNetPtr == 0) {
                Log.e(TAG, "Failed to load detector model");
                Toast.makeText(this, "目标检测模型加载失败", Toast.LENGTH_SHORT).show();
            } else {
                Log.i(TAG, "Detector model loaded successfully");
            }

            // 初始化检测结果容器
            detectedBoxes = new MatOfRect();
            detectedClassIds = new MatOfInt();

        } catch (Exception e) {
            Log.e(TAG, "Error initializing detector: " + e.getMessage());
            Toast.makeText(this, "目标检测初始化失败: " + e.getMessage(), Toast.LENGTH_SHORT).show();
        }
    }

    /**
     * 将assets中的文件复制到缓存目录
     */
    private String copyAssetToCache(String assetName) throws IOException {
        File cacheFile = new File(getCacheDir(), assetName);

        if (!cacheFile.exists()) {
            InputStream is = getAssets().open(assetName);
            FileOutputStream fos = new FileOutputStream(cacheFile);

            byte[] buffer = new byte[1024];
            int length;
            while ((length = is.read(buffer)) > 0) {
                fos.write(buffer, 0, length);
            }

            fos.close();
            is.close();
            Log.i(TAG, "Asset copied to cache: " + assetName);
        }

        return cacheFile.getAbsolutePath();
    }

    /**
     * 初始化按钮
     */
    @SuppressLint("SetTextI18n")
    private void initializeButtons() {
        btnEnhance = findViewById(R.id.button1);
        btnDehaze = findViewById(R.id.button2);
        btnCLAHE = findViewById(R.id.button3);
        btnMSRCR = findViewById(R.id.button4);
        btnAutoMode = findViewById(R.id.buttonAutoMode);
        btnMotionDetect = findViewById(R.id.buttonMotionDetect);   // 新增
        btnObjectDetect = findViewById(R.id.buttonObjectDetect);   // 新增

        setupButton(btnEnhance, "Enhance", ProcessingMode.ENHANCE);
        setupButton(btnDehaze, "Dehaze", ProcessingMode.DEHAZE);
        setupButton(btnCLAHE, "CLAHE", ProcessingMode.CLAHE);
        setupButton(btnMSRCR, "MSRCR", ProcessingMode.MSRCR);
        setupButton(btnMotionDetect, "运动检测", ProcessingMode.MOTION_DETECT);  // 新增
        setupButton(btnObjectDetect, "目标检测", ProcessingMode.OBJECT_DETECT);  // 新增

        // 自动模式按钮点击事件
        btnAutoMode.setOnClickListener(v -> toggleAutoMode());
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

    private void toggleAutoMode() {
        isAutoModeEnabled = !isAutoModeEnabled;
        String mode = isAutoModeEnabled ? "开启" : "关闭";
        Toast.makeText(this, "自动模式" + mode, Toast.LENGTH_SHORT).show();
        currentMode = ProcessingMode.NONE;  // 自动模式下不选择任何增强模式
        isFirstFrame = true;
        updateButtonStates();
    }

    private void updateButtonStates() {
        btnEnhance.setSelected(currentMode == ProcessingMode.ENHANCE);
        btnDehaze.setSelected(currentMode == ProcessingMode.DEHAZE);
        btnCLAHE.setSelected(currentMode == ProcessingMode.CLAHE);
        btnMSRCR.setSelected(currentMode == ProcessingMode.MSRCR);
        btnMotionDetect.setSelected(currentMode == ProcessingMode.MOTION_DETECT);  // 新增
        btnObjectDetect.setSelected(currentMode == ProcessingMode.OBJECT_DETECT);  // 新增
        btnAutoMode.setText(isAutoModeEnabled ? "关闭自动模式" : "开启自动模式");
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NotNull String[] permissions,
                                           @NotNull int[] grantResults) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults);

        if (requestCode == CAMERA_PERMISSION_REQUEST) {
            if (grantResults.length > 0 && grantResults[0] == PackageManager.PERMISSION_GRANTED) {
                mOpenCvCameraView.setCameraPermissionGranted();
                Log.i(TAG, "Camera permission granted");
            } else {
                String message = "Camera permission was not granted";
                Log.e(TAG, message);
                Toast.makeText(this, message, Toast.LENGTH_LONG).show();
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
            // 始终进行稳像处理
            processStab(inputFrame);

            if (isAutoModeEnabled) {
                processAutoMode(inputFrame);  // 自动模式处理
            } else {
                processFrame(inputFrame);  // 手动模式处理
            }

        } catch (Exception e) {
            Log.e(TAG, "Error processing frame: " + e.getMessage());
        }
        return inputFrame;
    }

    /**
     * 自动模式处理（根据亮度和运动状态选择处理方法）
     */
    private void processAutoMode(Mat frame) {
        // 获取图像的亮度（BGR通道的平均值）
        Scalar meanVal = Core.mean(frame);
        double brightness = meanVal.val[0];  // 亮度（BGR的平均值）

        // 判断设备是否在运动
        if (isDeviceMoving()) {
            // 设备在运动时进行视频稳像（已在onCameraFrame中处理）
            Log.d(TAG, "Device is moving, stabilization active");
        } else {
            // 设备静止时，根据图像亮度选择增强方法
            if (brightness < 50) {
                processEnhance(frame);  // 使用增强
            } else if (brightness > 200) {
                processCLAHE(frame);  // 使用CLAHE
            } else {
                processDehaze(frame);  // 使用去雾
            }
        }

        // 可选：调用 autoProcess 本地方法（如果需要native层的自动处理）
        // long matAddr1 = frame.getNativeObjAddr();
        // long matAddr2 = frame.getNativeObjAddr();
        // NativeProcessor.autoProcess(matAddr1, matAddr2, accelX, accelY, accelZ);
    }

    /**
     * 判断设备是否在运动
     */
    private boolean isDeviceMoving() {
        double acceleration = Math.sqrt(accelX * accelX + accelY * accelY + accelZ * accelZ);
        return acceleration > 12.0;  // 阈值调整（考虑重力加速度约9.8m/s²）
    }

    /**
     * 手动模式处理（根据选择的模式处理）
     */
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
            case MOTION_DETECT:      // 新增
                processMotionDetect(frame);
                break;
            case OBJECT_DETECT:      // 新增
                processObjectDetect(frame);
                break;
            default:
                break;
        }
    }

    // ==================== 原有的处理方法 ====================

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
        old.release();
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
        old.release();
    }

    // ==================== 新增：运动检测处理 ====================

    /**
     * 运动检测处理
     */
    private void processMotionDetect(Mat frame) {
        if (frame == null || frame.empty()) {
            return;
        }

        // 第一帧只保存，不处理
        if (isFirstFrame || lastFrameForMotion == null || lastFrameForMotion.empty()) {
            if (lastFrameForMotion == null) {
                lastFrameForMotion = new Mat();
            }
            frame.copyTo(lastFrameForMotion);
            isFirstFrame = false;
            return;
        }

        try {
            // 创建输出容器
            MatOfRect motionBoxes = new MatOfRect();

            // 调用native方法进行运动检测
            NativeProcessor.detectMotion(
                    lastFrameForMotion.getNativeObjAddr(),
                    frame.getNativeObjAddr(),
                    motionBoxes.getNativeObjAddr()
            );

            // 绘制检测到的运动区域
            Rect[] boxes = motionBoxes.toArray();
            for (Rect box : boxes) {
                // 绘制绿色边框
                Imgproc.rectangle(frame, box.tl(), box.br(),
                        new Scalar(0, 255, 0, 255), 3);

                // 添加文本标签
                Imgproc.putText(frame, "Motion",
                        new Point(box.x, box.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.6, new Scalar(0, 255, 0, 255), 2);
            }

            // 在屏幕上显示检测到的运动对象数量
            String countText = "Motion Objects: " + boxes.length;
            Imgproc.putText(frame, countText,
                    new Point(10, 30),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.8, new Scalar(0, 255, 0, 255), 2);

            // 更新上一帧
            frame.copyTo(lastFrameForMotion);

            // 释放临时Mat
            motionBoxes.release();

        } catch (Exception e) {
            Log.e(TAG, "Error in motion detection: " + e.getMessage());
        }
    }

    // ==================== 新增：目标检测处理 ====================

    /**
     * 目标检测处理
     */
    private void processObjectDetect(Mat frame) {
        if (frame == null || frame.empty() || detectorNetPtr == 0) {
            if (detectorNetPtr == 0) {
                Log.w(TAG, "Detector model not loaded");
            }
            return;
        }

        try {
            // 调用native方法进行目标检测
            NativeProcessor.detectObjectsNN(
                    detectorNetPtr,
                    frame.getNativeObjAddr(),
                    detectedBoxes.getNativeObjAddr(),
                    detectedClassIds.getNativeObjAddr(),
                    DETECTION_CONFIDENCE
            );

            // 绘制检测结果
            Rect[] boxes = detectedBoxes.toArray();
            int[] classIds = detectedClassIds.toArray();

            for (int i = 0; i < boxes.length; i++) {
                Rect box = boxes[i];
                int classId = classIds[i];

                // 绘制蓝色边框
                Imgproc.rectangle(frame, box.tl(), box.br(),
                        new Scalar(255, 0, 0, 255), 3);

                // 添加类别标签
                String label = getClassLabel(classId);
                Imgproc.putText(frame, label,
                        new Point(box.x, box.y - 10),
                        Imgproc.FONT_HERSHEY_SIMPLEX,
                        0.6, new Scalar(255, 0, 0, 255), 2);
            }

            // 在屏幕上显示检测到的对象数量
            String countText = "Detected Objects: " + boxes.length;
            Imgproc.putText(frame, countText,
                    new Point(10, 30),
                    Imgproc.FONT_HERSHEY_SIMPLEX,
                    0.8, new Scalar(255, 0, 0, 255), 2);

        } catch (Exception e) {
            Log.e(TAG, "Error in object detection: " + e.getMessage());
        }
    }

    /**
     * 获取类别标签（MobileNet SSD的COCO类别）
     */
    private String getClassLabel(int classId) {
        String[] labels = {
                "background", "aeroplane", "bicycle", "bird", "boat",
                "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                "sofa", "train", "tvmonitor"
        };

        if (classId >= 0 && classId < labels.length) {
            return labels[classId];
        }
        return "unknown";
    }

    // ==================== 生命周期回调 ====================

    @Override
    public void onCameraViewStarted(int width, int height) {
        Log.i(TAG, "Camera view started: " + width + "x" + height);
    }

    @Override
    public void onCameraViewStopped() {
        Log.i(TAG, "Camera view stopped");
        releaseResources();
    }

    @Override
    public void onPause() {
        super.onPause();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
        Log.i(TAG, "Activity paused");
    }

    @Override
    public void onResume() {
        super.onResume();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.enableView();
        }
        isFirstFrame = true;
        currentMode = ProcessingMode.NONE;
        updateButtonStates();
        Log.i(TAG, "Activity resumed");
    }

    @Override
    public void onDestroy() {
        super.onDestroy();

        // 停止相机
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }

        // 释放传感器
        if (sensorManager != null) {
            sensorManager.unregisterListener(sensorEventListener);
            Log.i(TAG, "Sensor listener unregistered");
        }

        // 释放检测器
        if (detectorNetPtr != 0) {
            NativeProcessor.releaseObjectDetector(detectorNetPtr);
            detectorNetPtr = 0;
            Log.i(TAG, "Object detector released");
        }

        // 释放其他资源
        releaseResources();

        Log.i(TAG, "Activity destroyed");
    }

    /**
     * 释放所有Mat资源
     */
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
            if (detectedBoxes != null) {
                detectedBoxes.release();
                detectedBoxes = null;
            }
            if (detectedClassIds != null) {
                detectedClassIds.release();
                detectedClassIds = null;
            }
            Log.i(TAG, "All Mat resources released");
        } catch (Exception e) {
            Log.e(TAG, "Error releasing resources: " + e.getMessage());
        }
    }
}