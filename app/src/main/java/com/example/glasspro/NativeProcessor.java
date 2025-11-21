package com.example.glasspro;

import org.opencv.core.Mat;
import org.opencv.core.MatOfInt;
import org.opencv.core.MatOfRect;
import org.opencv.core.Point;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Iterator;
import java.util.List;

public class NativeProcessor {

    // ==================== 本地方法声明 (保持不变) ====================
    public static native void enhance(long matAddr1, long matAddr2, double noiseLevel);
    public static native void dehaze(long matAddr);
    public static native void enhanceByCLAHE(long matAddr);
    public static native void enhanceByMSRCR(long matAddr);
    public static native void videoStab(long matAddr1, long matAddr2);
    public static native void detectMotion(long matAddrPrev, long matAddrCurrent, long matAddrOutBoxes);
    public static native long loadObjectDetector(String proto, String model);
    public static native void releaseObjectDetector(long netPtr);
    public static native void detectObjectsNN(long netPtr, long frameAddr, long boxesAddr, long classIdsAddr, float confThreshold);

    // =========================================================
    //        优化后的 Java 层逻辑
    // =========================================================

    // 1. 类别与颜色定义
    private static final List<String> CLASS_NAMES = Arrays.asList(
            "background", "aeroplane", "bicycle", "bird", "boat", "bottle",
            "bus", "car", "cat", "chair", "cow", "diningtable", "dog",
            "horse", "motorbike", "person", "pottedplant", "sheep", "sofa",
            "train", "tvmonitor"
    );

    private static final Scalar[] CLASS_COLORS = new Scalar[]{
            new Scalar(0, 0, 0), new Scalar(255, 0, 0), new Scalar(0, 255, 0),
            new Scalar(0, 0, 255), new Scalar(255, 255, 0), new Scalar(0, 255, 255),
            new Scalar(255, 0, 255), new Scalar(192, 192, 192), new Scalar(128, 128, 128),
            new Scalar(128, 0, 0), new Scalar(128, 128, 0), new Scalar(0, 128, 0),
            new Scalar(128, 0, 128), new Scalar(0, 128, 128), new Scalar(0, 0, 128),
            new Scalar(255, 165, 0), new Scalar(255, 255, 255), new Scalar(128, 128, 128),
            new Scalar(0, 0, 0), new Scalar(0, 0, 0), new Scalar(0, 0, 0)
    };

    // --- [优化点 1] 调整参数：更严格的过滤，更快的响应 ---

    // 提高平滑灵敏度：原来是 0.6 (比较粘手)，现在改为 0.7 (更跟手)
    private static final float SMOOTH_FACTOR = 0.7f;

    // 降低容错帧数：原来是 5，改为 2。
    // 移动时，如果物体真的消失了，保留太久会造成框在空中飘的假象。
    private static final int MAX_MISSED_FRAMES = 2;

    // 匹配距离阈值
    private static final double MAX_MATCH_DIST = 200.0;

    // [新增] 瞬间移动阈值：如果新旧框中心距离超过 50 像素，说明运动很快，直接跳过去，不进行平滑
    private static final double SNAP_DISTANCE_THRESHOLD = 50.0;

    // 3. 内部状态管理
    private static final List<StableBox> stableBoxes = new ArrayList<>();

    private static class StableBox {
        Rect rect;
        int classId;
        int missedCount;

        StableBox(Rect rect, int classId) {
            this.rect = rect.clone();
            this.classId = classId;
            this.missedCount = 0;
        }
    }

    /**
     * 核心业务方法
     */
    public static void detectAndDraw(Mat frame, long netPtr, MatOfRect bufferBoxes, MatOfInt bufferIds, float inputConfidence) {
        if (netPtr == 0 || frame.empty()) return;

        // --- [优化点 2] 强制提高置信度阈值 ---
        // 即使 MainActivity 传进来 0.3，我们在这里也强制过滤掉低于 0.55 的垃圾框。
        // 移动模糊会产生大量 0.3~0.4 的误识别。
        float effectiveConfidence = Math.max(inputConfidence, 0.55f);

        // A. C++ 推理
        bufferBoxes.release();
        bufferIds.release();
        detectObjectsNN(netPtr, frame.getNativeObjAddr(), bufferBoxes.getNativeObjAddr(), bufferIds.getNativeObjAddr(), effectiveConfidence);

        Rect[] rawBoxes = bufferBoxes.toArray();
        int[] rawIds = bufferIds.toArray();

        // B. 防抖逻辑 (已优化)
        updateStabilizer(rawBoxes, rawIds);

        // C. 绘制
        drawStart(frame);
    }

    // 私有：更新稳定器状态
    private static void updateStabilizer(Rect[] rawBoxes, int[] rawIds) {
        if (rawBoxes == null || rawIds == null) return;
        boolean[] rawUsed = new boolean[rawBoxes.length];

        Iterator<StableBox> it = stableBoxes.iterator();
        while (it.hasNext()) {
            StableBox sb = it.next();
            int bestMatchIndex = -1;
            double minDistance = Double.MAX_VALUE;

            // 寻找最佳匹配
            for (int i = 0; i < rawBoxes.length; i++) {
                if (rawUsed[i]) continue;
                if (rawIds[i] != sb.classId) continue;

                double dist = getDistance(sb.rect, rawBoxes[i]);
                if (dist < minDistance) {
                    minDistance = dist;
                    bestMatchIndex = i;
                }
            }

            if (bestMatchIndex != -1 && minDistance < MAX_MATCH_DIST) {
                Rect raw = rawBoxes[bestMatchIndex];

                // --- [优化点 3] 动态平滑逻辑 ---
                if (minDistance > SNAP_DISTANCE_THRESHOLD) {
                    // 距离太远（快速移动），直接“瞬移”过去，消除滞后感
                    sb.rect = raw.clone();
                } else {
                    // 距离较近（微小抖动），启用加权平滑
                    sb.rect.x = (int) (raw.x * SMOOTH_FACTOR + sb.rect.x * (1 - SMOOTH_FACTOR));
                    sb.rect.y = (int) (raw.y * SMOOTH_FACTOR + sb.rect.y * (1 - SMOOTH_FACTOR));
                    sb.rect.width = (int) (raw.width * SMOOTH_FACTOR + sb.rect.width * (1 - SMOOTH_FACTOR));
                    sb.rect.height = (int) (raw.height * SMOOTH_FACTOR + sb.rect.height * (1 - SMOOTH_FACTOR));
                }

                sb.missedCount = 0;
                rawUsed[bestMatchIndex] = true;
            } else {
                sb.missedCount++;
            }

            if (sb.missedCount > MAX_MISSED_FRAMES) {
                it.remove();
            }
        }

        // 添加新框
        for (int i = 0; i < rawBoxes.length; i++) {
            if (!rawUsed[i]) {
                stableBoxes.add(new StableBox(rawBoxes[i], rawIds[i]));
            }
        }
    }

    private static void drawStart(Mat frame) {
        for (StableBox sb : stableBoxes) {
            if (sb.classId < 0 || sb.classId >= CLASS_NAMES.size()) continue;

            String label = CLASS_NAMES.get(sb.classId);
            Scalar color = CLASS_COLORS[sb.classId];

            Imgproc.rectangle(frame, sb.rect, color, 2);

            Point labelPos = new Point(sb.rect.x, sb.rect.y - 5);
            if (labelPos.y < 20) labelPos.y = sb.rect.y + 20;
            Imgproc.putText(frame, label, labelPos, Imgproc.FONT_HERSHEY_SIMPLEX, 0.6, new Scalar(255, 255, 255), 2);
        }
    }

    private static double getDistance(Rect r1, Rect r2) {
        double cx1 = r1.x + r1.width / 2.0;
        double cy1 = r1.y + r1.height / 2.0;
        double cx2 = r2.x + r2.width / 2.0;
        double cy2 = r2.y + r2.height / 2.0;
        return Math.sqrt(Math.pow(cx1 - cx2, 2) + Math.pow(cy1 - cy2, 2));
    }

    public static void resetStabilizer() {
        stableBoxes.clear();
    }
}