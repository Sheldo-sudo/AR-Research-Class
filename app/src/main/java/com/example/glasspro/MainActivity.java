package com.example.glasspro;

// 静态导入新的 NativeProcessor
import static com.example.glasspro.NativeProcessor.dehaze;
import static com.example.glasspro.NativeProcessor.enhance;
import static com.example.glasspro.NativeProcessor.enhanceByCLAHE;
import static com.example.glasspro.NativeProcessor.enhanceByMSRCR;
import static com.example.glasspro.NativeProcessor.videoStab;
import static com.example.glasspro.NativeProcessor.loadObjectDetector;
import static com.example.glasspro.NativeProcessor.releaseObjectDetector;
import static com.example.glasspro.NativeProcessor.detectObjectsNN;

import android.Manifest;
import android.annotation.SuppressLint;
import android.app.Activity;
import android.content.Context;
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
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;

import java.io.File;
import java.io.FileOutputStream;
import java.io.InputStream;
import java.io.OutputStream;
import java.util.Arrays;
import java.util.List;


public class MainActivity extends Activity implements CvCameraViewListener2 {
    private static final String TAG = "MainActivity";

    // --- [!! 关键修复：加载所有 C++ 库 !!] ---
    static {
        if (!OpenCVLoader.initDebug()) {
            Log.e(TAG, "STATIC: OpenCV library load failed!");
        } else {
            Log.d(TAG, "STATIC: OpenCV library loaded successfully!");
        }

        try {
            // 加载在 CMakeLists.txt 中定义的 *所有* 库
            System.loadLibrary("dehaze");
            System.loadLibrary("enhance");
            System.loadLibrary("threshold");
            System.loadLibrary("stab");           // <-- 包含 videoStab()
            System.loadLibrary("motion");
            System.loadLibrary("native-lib");     // <-- 你的主库
            System.loadLibrary("vision_processor"); // <-- 包含 DNN 函数

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
    // private Mat lastFrameForMotion; // 不再需要
    private Mat lastFrameForEnhance;
    // UI组件
    private Button btnEnhance;
    private Button btnDehaze;
    private Button btnCLAHE;
    private Button btnMSRCR;

    // --- [!! 调试：DNN 目标检测变量 !!] ---
    private long dnnNetPtr = 0;
    private static final String MODEL_PROTO = "mobilenet_ssd.prototxt";
    private static final String MODEL_WEIGHTS = "mobilenet_ssd.caffemodel";

    // [!! 修复：降低置信度阈值 !!]
    private static final float CONFIDENCE_THRESHOLD = 0.2f; // 40% 降到 10%
    // --- [!! 修复结束 !!] ---

    // MobileNet-SSD COCO 数据集的前21个类别
    private final List<String> classNames = Arrays.asList(
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor");
    private final Scalar[] classColors = new Scalar[]{
            new Scalar(0, 0, 0), new Scalar(255, 0, 0), new Scalar(0, 255, 0),
            new Scalar(0, 0, 255), new Scalar(255, 255, 0), new Scalar(0, 255, 255),
            new Scalar(255, 0, 255), new Scalar(192, 192, 192), new Scalar(128, 128, 128),
            new Scalar(128, 0, 0), new Scalar(128, 128, 0), new Scalar(0, 128, 0),
            new Scalar(128, 0, 128), new Scalar(0, 128, 128), new Scalar(0, 0, 128),
            new Scalar(255, 165, 0), // person (橙色)
            new Scalar(255, 255, 255), new Scalar(128, 128, 128), new Scalar(0, 0, 0),
            new Scalar(0, 0, 0), new Scalar(0, 0, 0)
    };


    private MatOfRect lastDetectedBoxes = null;
    private MatOfInt lastDetectedClassIds = null;
    private Rect[] lastBoxesArray = new Rect[0];
    private int[] lastIdsArray = new int[0];


    @Override
    public void onCreate(Bundle savedInstanceState) {
        Log.i(TAG, "called onCreate");
        super.onCreate(savedInstanceState);

        if (!initializeOpenCV()) {
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
    @SuppressLint("SetTextI1n")
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
            if (lastDetectedBoxes == null) {
                lastDetectedBoxes = new MatOfRect();
                Log.d(TAG, "lastDetectedBoxes created lazily on first frame.");
            }
            if (lastDetectedClassIds == null) {
                lastDetectedClassIds = new MatOfInt();
                Log.d(TAG, "lastDetectedClassIds created lazily on first frame.");
            }

            // 1. 稳像处理 (每帧运行, 全新 C++ 实现)
            processStab(inputFrame);

            // 2. 按键增强处理 (每帧运行)
            processFrame(inputFrame);

            // 3. 自动的、每帧的 DNN 目标 "检测" (昂贵)
            processObjectDetection(inputFrame);

            // 4. 自动的、每帧的目标 "绘制" (廉价)
            drawDetections(inputFrame);

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
     * 在后台自动运行的、每帧的 DNN "检测" (昂贵)
     */
    private void processObjectDetection(Mat frame) {
        if (dnnNetPtr != 0 && lastDetectedBoxes != null && lastDetectedClassIds != null) {
            lastDetectedBoxes.release();
            lastDetectedClassIds.release();
            lastDetectedBoxes = new MatOfRect();
            lastDetectedClassIds = new MatOfInt();

            // 调用 Native DNN 检测
            detectObjectsNN(
                    dnnNetPtr,
                    frame.getNativeObjAddr(),
                    lastDetectedBoxes.getNativeObjAddr(),
                    lastDetectedClassIds.getNativeObjAddr(),
                    CONFIDENCE_THRESHOLD // [!! 修复 !!] 现在传入的是 0.1f
            );

            lastBoxesArray = lastDetectedBoxes.toArray();
            lastIdsArray = lastDetectedClassIds.toArray();
        }
    }


    /**
     * [!! 调试版本 !!]
     * 在 "每帧" 运行的目标 "绘制" (廉价)
     */
    private void drawDetections(Mat frame) {
        if (lastBoxesArray != null && lastIdsArray != null && lastBoxesArray.length > 0 && lastBoxesArray.length == lastIdsArray.length) {

            // [!! 新增调试日志 !!] 告诉我们 C++ 返回了多少个物体
            Log.d(TAG, "Java drawDetections: C++ returned " + lastBoxesArray.length + " objects this frame.");

            for (int i = 0; i < lastBoxesArray.length; i++) {
                Rect box = lastBoxesArray[i];
                int classId = lastIdsArray[i];

                String label = (classId >= 0 && classId < classNames.size()) ? classNames.get(classId) : "Unknown";
                Scalar color = (classId >= 0 && classId < classColors.length) ? classColors[classId] : new Scalar(255, 255, 255);

                // [!! 新增调试日志 !!] 告诉我们每个物体的具体标签
                Log.d(TAG, "  -> Drawing Object " + i + ": " + label);

                // [!! 关键修改：注释掉过滤器 !!]
                // if (label.equals("person") || label.equals("car") || label.equals("bus") || label.equals("motorbike")) {

                Imgproc.rectangle(frame, box, color, 3);
                Point labelPos = new Point(box.x, box.y - 10);
                if (labelPos.y < 10) {
                    labelPos.y = box.y + 30;
                }
                Imgproc.putText(frame, label, labelPos, Imgproc.FONT_HERSHEY_SIMPLEX, 1.0, color, 3);

                // } // [!! 关键修改：注释掉过滤器 !!]
            }
        }
        // [!! 新增调试日志 !!]
        else if (lastBoxesArray != null && lastBoxesArray.length == 0) {
            Log.d(TAG, "Java drawDetections: C++ returned 0 objects.");
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

    /**
     * 稳像处理
     */
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
        isFirstFrame = true;
        if (lastFrameForStab != null) { lastFrameForStab.release(); lastFrameForStab = null; }
        if (lastFrameForEnhance != null) { lastFrameForEnhance.release(); lastFrameForEnhance = null; }
        lastBoxesArray = new Rect[0];
        lastIdsArray = new int[0];

        // --- [!! 新增：加载 DNN 网络 !!] ---
        if (dnnNetPtr == 0) {
            try {
                String protoPath = getPathFromAsset(this, MODEL_PROTO);
                String modelPath = getPathFromAsset(this, MODEL_WEIGHTS);

                if (protoPath == null || modelPath == null) {
                    Log.e(TAG, "Failed to get model files from assets");
                    Toast.makeText(this, "Failed to load models", Toast.LENGTH_LONG).show();
                    return;
                }

                Log.d(TAG, "Loading DNN Model...");
                Log.d(TAG, "Proto: " + protoPath);
                Log.d(TAG, "Model: " + modelPath);

                dnnNetPtr = loadObjectDetector(protoPath, modelPath);

                if (dnnNetPtr == 0) {
                    Log.e(TAG, "loadObjectDetector returned 0");
                    Toast.makeText(this, "Failed to load DNN native", Toast.LENGTH_LONG).show();
                } else {
                    Log.i(TAG, "DNN Model loaded successfully. Pointer: " + dnnNetPtr);
                }
            } catch (Exception e) {
                Log.e(TAG, "Error loading DNN model: " + e.getMessage());
            }
        }
        // --- [!! 加载结束 !!] ---
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
        currentMode = ProcessingMode.NONE;
        updateButtonStates();
    }

    @Override
    public void onDestroy() {
        super.onDestroy();
        if (mOpenCvCameraView != null) {
            mOpenCvCameraView.disableView();
        }
        releaseResources(); // 确保在 onDestroy 中也释放
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

            if (lastDetectedBoxes != null) {
                lastDetectedBoxes.release();
                lastDetectedBoxes = null;
            }
            if (lastDetectedClassIds != null) {
                lastDetectedClassIds.release();
                lastDetectedClassIds = null;
            }

            lastBoxesArray = new Rect[0];
            lastIdsArray = new int[0];

            // --- [!! 新增：释放 DNN 网络 !!] ---
            if (dnnNetPtr != 0) {
                Log.d(TAG, "Releasing DNN object detector...");
                releaseObjectDetector(dnnNetPtr);
                dnnNetPtr = 0;
            }

        } catch (Exception e) {
            Log.e(TAG, "Error releasing resources: " + e.getMessage());
        }
    }


    /**
     * [!! 新增辅助函数 !!]
     * 将 Assets 中的文件复制到 App 的内部存储，并返回其绝对路径。
     */
    private String getPathFromAsset(Context context, String assetName) {
        File outFile = new File(context.getCacheDir(), assetName);
        if (!outFile.exists()) {
            Log.d(TAG, "Asset file " + assetName + " not in cache. Copying...");
            try (InputStream in = context.getAssets().open(assetName);
                 OutputStream out = new FileOutputStream(outFile)) {

                byte[] buffer = new byte[4096];
                int read;
                while ((read = in.read(buffer)) != -1) {
                    out.write(buffer, 0, read);
                }
                Log.d(TAG, "Asset file copied successfully to: " + outFile.getAbsolutePath());
            } catch (Exception e) {
                Log.e(TAG, "Failed to copy asset file: " + assetName, e);
                return null;
            }
        } else {
            Log.d(TAG, "Asset file " + assetName + " already in cache.");
        }

        return outFile.getAbsolutePath();
    }
}