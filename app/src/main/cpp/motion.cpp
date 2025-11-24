#include "motion.h"
#include "videoStab.h"
#include <jni.h>

/**
 * @brief 检测运动物体，并将轮廓的边界框存入 out_boxes。
 * (这现在是纯 C++ 逻辑，不含 JNI)
 */
void detectMotion(const cv::Mat& prev_frame_rgba,
                  const cv::Mat& current_frame_rgba,
                  std::vector<cv::Rect>& out_boxes) { // <-- [!! 签名大修改 !!]

    // --- 1. 稳像处理 ---
    // (保持不变) Java层已经处理了稳像

    // --- 2. 准备 "当前帧" 灰度图 ---
    cv::Mat bgr_frame, current_gray;
    cv::cvtColor(current_frame_rgba, bgr_frame, cv::COLOR_RGBA2BGR);
    cv::cvtColor(bgr_frame, current_gray, cv::COLOR_BGR2GRAY);

    // --- [!! 调优 1：解决“手机抖动”!!] ---
    cv::GaussianBlur(current_gray, current_gray, cv::Size(21, 21), 0);

    // --- 3. 准备 "上一帧" 灰度图 ---
    cv::Mat prev_bgr_frame, prev_gray;
    cv::cvtColor(prev_frame_rgba, prev_bgr_frame, cv::COLOR_RGBA2BGR);
    cv::cvtColor(prev_bgr_frame, prev_gray, cv::COLOR_BGR2GRAY);
    cv::GaussianBlur(prev_gray, prev_gray, cv::Size(21, 21), 0);

    // --- 4. 核心算法：帧差法 ---
    cv::Mat frame_delta;
    cv::absdiff(prev_gray, current_gray, frame_delta);

    // --- [!! 调优 1：解决“手机抖动”!!] ---
    cv::Mat thresh;
    cv::threshold(frame_delta, thresh, 50, 255, cv::THRESH_BINARY);

    // --- [!! 调优 2：解决“很多个框”!!] ---
    cv::dilate(thresh, thresh, cv::Mat(), cv::Point(-1, -1), 10);

    // 7. 查找轮廓
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(thresh, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_SIMPLE);

    // --- [!! 关键修改 !!] ---
    // 清空上一帧的方框
    out_boxes.clear();

    // 8. 遍历轮廓并 "返回" 方框
    for (const auto& contour : contours) {

        // 9. 面积过滤
        if (cv::contourArea(contour) < 3000) {
            continue;
        }

        cv::Rect bounding_box = cv::boundingRect(contour);

        // [!! 删除 !!] 我们不再在C++中绘制
        // cv::rectangle(current_frame_rgba, bounding_box, cv::Scalar(0, 255, 0, 255), 3);

        // [!! 新增 !!] 我们把方框 "返回" 给调用者
        out_boxes.push_back(bounding_box);
    }
}


// --- JNI 接口函数 ---
// [!! 关键修复 !!] 我们的JNI函数现在接收第三个 long (用于输出 MatOfRect)
extern "C"
JNIEXPORT void JNICALL
Java_com_example_glasspro_NativeProcessor_detectMotion(
        JNIEnv *env,
        jclass clazz,
        jlong matAddrPrev,      // 上一帧
        jlong matAddrCurrent,   // 当前帧
        jlong matAddrOutBoxes   // [!! 新增 !!] 输出方框(MatOfRect)的地址
) {

    cv::Mat &prev_frame = *(cv::Mat *) matAddrPrev;
    cv::Mat &current_frame = *(cv::Mat *) matAddrCurrent;
    cv::Mat &out_boxes_mat = *(cv::Mat *) matAddrOutBoxes; // <-- [!! 新增 !!]

    if (prev_frame.empty() || current_frame.empty()) {
        return;
    }

    // 1. 创建一个临时的 C++ vector 来存储结果
    std::vector<cv::Rect> detected_boxes;

    // 2. 调用我们的 C++ 核心函数
    detectMotion(prev_frame, current_frame, detected_boxes);

    // 3. 将 C++ vector 转换为 Java 的 MatOfRect (OpenCV 标准做法)
    // 检查 detected_boxes 是否有内容
    if (detected_boxes.empty()) {
        out_boxes_mat.release();
    } else {
        // 使用 fromList 将 C++ vector 转换为 OpenCV Mat
        // (注意：OpenCV for Android 的 MatOfRect 在 C++ 层就是一个 Mat)
        out_boxes_mat.create(detected_boxes.size(), 1, CV_32SC4); // 4个 int (x, y, w, h)
        for (size_t i = 0; i < detected_boxes.size(); ++i) {
            out_boxes_mat.at<cv::Vec4i>(i, 0) = cv::Vec4i(
                    detected_boxes[i].x,
                    detected_boxes[i].y,
                    detected_boxes[i].width,
                    detected_boxes[i].height
            );
        }
    }
}