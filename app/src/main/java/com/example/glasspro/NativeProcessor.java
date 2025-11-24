package com.example.glasspro;

public class NativeProcessor {
    static {
        try {
            System.loadLibrary("dehaze");
            System.loadLibrary("enhance");
            System.loadLibrary("threshold");
            System.loadLibrary("stab");
            // 新增：运动检测，目标检测
            System.loadLibrary("motion");
            System.loadLibrary("vision");
        } catch (UnsatisfiedLinkError e) {
            throw new RuntimeException("Native lib load failed: " + e.getMessage());
        }
    }
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
     * Native方法：自动图像处理（选择合适的图像增强或去雾方法）
     * @param matAddr1 图像1的内存地址
     * @param matAddr2 图像2的内存地址
     * @param accelX X轴加速度
     * @param accelY Y轴加速度
     * @param accelZ Z轴加速度
     */
    public static native void autoProcess(long matAddr1, long matAddr2, float accelX, float accelY, float accelZ);
    //=========== 运动检测 ===========
    /**
     * Native方法：检测运动物体
     * @param matAddrPrev 上一帧的内存地址
     * @param matAddrCurrent 当前帧的内存地址
     * @param matAddrOutBoxes 输出边界框的MatOfRect地址
     */
    public static native void detectMotion(long matAddrPrev, long matAddrCurrent, long matAddrOutBoxes);
    // ==================== 新增：目标检测 ====================
    /**
     * Native方法：加载目标检测模型
     * @param protoPath 模型配置文件路径
     * @param modelPath 模型权重文件路径
     * @return 模型指针
     */
    public static native long loadObjectDetector(String protoPath, String modelPath);

    /**
     * Native方法：释放目标检测模型
     * @param netPtr 模型指针
     */
    public static native void releaseObjectDetector(long netPtr);

    /**
     * Native方法：使用神经网络检测目标
     * @param netPtr 模型指针
     * @param frameAddr 输入图像地址
     * @param boxesAddr 输出边界框地址
     * @param classIdsAddr 输出类别ID地址
     * @param confThreshold 置信度阈值
     */
    public static native void detectObjectsNN(long netPtr, long frameAddr, long boxesAddr,
                                              long classIdsAddr, float confThreshold);
}
