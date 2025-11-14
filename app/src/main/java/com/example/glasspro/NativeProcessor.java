package com.example.glasspro;

public class NativeProcessor {

    // --- [!! 关键修复 !!] ---
    //
    // 删除了这里所有的 static { ... } 代码块。
    // 这个代码块是导致闪退的根源，因为它在 OpenCV 加载之前运行。
    // 我们将把所有库的加载，全部移到 MainActivity.java 中。
    //
    // --- [!! 修复结束 !!] ---


    // ==================== 本地方法声明 ====================
    /**
     * Native方法：增强图像。
     * @param matAddr1 图像1的内存地址
     * @param matAddr2 图像2的内存地址
     * @param noiseLevel 噪声水平
     */
    public static native void enhance(long matAddr1, long matAddr2, double noiseLevel);
    /**
     * Native方法：去雾处理。
     * @param matAddr 图像的内存地址
     */
    public static native void dehaze(long matAddr);
    /**
     * Native方法：使用CLAHE增强图像。
     * @param matAddr 图像的内存地址
     */
    public static native void enhanceByCLAHE(long matAddr);
    /**
     * Native方法：使用MSRCR增强图像。
     * @param matAddr 图像的内存地址
     */
    public static native void enhanceByMSRCR(long matAddr);
    /**
     * Native方法：稳定图像。
     * @param matAddr1 图像1的内存地址
     * @param matAddr2 图像2的内存地址
     */
    public static  native void videoStab(long matAddr1, long matAddr2);

    /**
     * Native 方法：使用帧差法检测运动。
     * [!! 关键修复 !!] 修改为接收三个 Mat 地址
     * @param matAddrPrev 上一帧的内存地址
     * @param matAddrCurrent 当前帧的内存地址
     * @param matAddrOutBoxes [输出] MatOfRect 的内存地址，用于 C++ 返回方框
     */
    public static native void detectMotion(long matAddrPrev, long matAddrCurrent, long matAddrOutBoxes);


    // --- [!! 以下是新增的 DNN 目标检测函数 !!] ---

    /**
     * [!! 新增 !!]
     * Native 方法：加载 DNN 目标检测网络。
     * @param proto 模型的 .prototxt 文件路径
     * @param model 模型的 .caffemodel 文件路径
     * @return 指向 C++ 中 cv::dnn::Net 对象的指针 (jlong)
     */
    public static native long loadObjectDetector(String proto, String model);

    /**
     * [!! 新增 !!]
     * Native 方法：释放 DNN 网络占用的内存。
     * @param netPtr loadObjectDetector 返回的指针
     */
    public static native void releaseObjectDetector(long netPtr);

    /**
     * [!! 新增 !!]
     * Native 方法：运行神经网络检测。
     * @param netPtr C++ 网络指针
     * @param frameAddr 当前帧的 Mat 地址
     * @param boxesAddr (输出) MatOfRect 的地址，用于存放检测框
     * @param classIdsAddr (输出) MatOfInt 的地址，用于存放类别 ID
     * @param confThreshold 置信度阈值 (例如 0.4)
     */
    public static native void detectObjectsNN(long netPtr, long frameAddr, long boxesAddr, long classIdsAddr, float confThreshold);
}