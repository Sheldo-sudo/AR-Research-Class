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
}
