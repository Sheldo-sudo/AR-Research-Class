package com.example.glasspro;

public class NativeProcessor {
    static {
        try {
            System.loadLibrary("dehaze");
            System.loadLibrary("enhance");
            System.loadLibrary("threshold");
            System.loadLibrary("stab");
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
}
