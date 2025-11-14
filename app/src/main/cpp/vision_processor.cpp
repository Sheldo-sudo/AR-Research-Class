#include "vision_processor.h"
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/dnn.hpp>
#include <vector>
#include <android/log.h>

// 修改 TAG 为更正式的名称
#define LOG_TAG "VisionProcessorNative"
#define LOGI(...) __android_log_print(ANDROID_LOG_INFO, LOG_TAG, __VA_ARGS__)
#define LOGE(...) __android_log_print(ANDROID_LOG_ERROR, LOG_TAG, __VA_ARGS__)

using namespace cv;
using namespace std;
using namespace cv::dnn;

/**
 * 帮助函数：将 jlong 转换为 Mat&
 */
static Mat &jlongToMat(jlong addr) {
    return *reinterpret_cast<Mat *>(addr);
}

JNIEXPORT jlong JNICALL
Java_com_example_glasspro_NativeProcessor_loadObjectDetector(JNIEnv *env, jclass clazz,
                                                             jstring proto, jstring model) {
    const char *proto_path = env->GetStringUTFChars(proto, 0);
    const char *model_path = env->GetStringUTFChars(model, 0);

    // 加载模型时保留日志是有益的，因为它只运行一次
    LOGI("Loading model from: %s", model_path);

    Net *net = new Net();
    try {
        *net = readNetFromCaffe(proto_path, model_path);
        net->setPreferableBackend(DNN_BACKEND_OPENCV);
        net->setPreferableTarget(DNN_TARGET_CPU);
        LOGI("DNN Model loaded successfully.");
    } catch (const cv::Exception &e) {
        LOGE("Failed to load DNN Model: %s", e.what());
        delete net;
        net = nullptr;
    }

    env->ReleaseStringUTFChars(proto, proto_path);
    env->ReleaseStringUTFChars(model, model_path);

    return reinterpret_cast<jlong>(net);
}

JNIEXPORT void JNICALL
Java_com_example_glasspro_NativeProcessor_releaseObjectDetector(JNIEnv *env, jclass clazz,
                                                                jlong net_ptr) {
    if (net_ptr != 0) {
        Net *net = reinterpret_cast<Net *>(net_ptr);
        delete net;
        LOGI("DNN Model released.");
    }
}

JNIEXPORT void JNICALL
Java_com_example_glasspro_NativeProcessor_detectObjectsNN(JNIEnv *env, jclass clazz,
                                                          jlong net_ptr, jlong frame_addr,
                                                          jlong boxes_addr, jlong class_ids_addr,
                                                          jfloat conf_threshold) {
    if (net_ptr == 0) {
        // 这种属于严重错误，建议保留 LOGE
        LOGE("net_ptr is 0, aborting detection.");
        return;
    }

    try {
        Net *net = reinterpret_cast<Net *>(net_ptr);
        Mat &frame = jlongToMat(frame_addr); // RGBA
        Mat &out_boxes_mat = jlongToMat(boxes_addr);
        Mat &out_ids_mat = jlongToMat(class_ids_addr);

        if (frame.empty()) {
            return;
        }

        // 1. 预处理：RGBA 转 BGR (逻辑保留)
        Mat frame_bgr;
        cvtColor(frame, frame_bgr, COLOR_RGBA2BGR);

        // 2. 创建 Blob (参数保留：300x300, mean=127.5, swapRB=false)
        Mat blob = blobFromImage(frame_bgr, 0.007843, Size(300, 300), Scalar(127.5, 127.5, 127.5), false, false);

        // 3. 前向传播
        net->setInput(blob);
        Mat detections = net->forward();

        // 4. 解析结果
        Mat detection_matrix(detections.size[2], detections.size[3], CV_32F, detections.data);

        vector<Rect> boxes_vec;
        vector<int> ids_vec;

        float frame_height = (float)frame.rows;
        float frame_width = (float)frame.cols;

        for (int i = 0; i < detection_matrix.rows; i++) {
            float confidence = detection_matrix.at<float>(i, 2);

            if (confidence > conf_threshold) {
                int class_id = static_cast<int>(detection_matrix.at<float>(i, 1));

                int x_left = static_cast<int>(detection_matrix.at<float>(i, 3) * frame_width);
                int y_top = static_cast<int>(detection_matrix.at<float>(i, 4) * frame_height);
                int x_right = static_cast<int>(detection_matrix.at<float>(i, 5) * frame_width);
                int y_bottom = static_cast<int>(detection_matrix.at<float>(i, 6) * frame_height);

                int w = x_right - x_left;
                int h = y_bottom - y_top;

                // 过滤逻辑 (逻辑保留：宽和高大于0，且面积 > 100，且在边界内)
                if (w > 0 && h > 0 && (w * h > 100) &&
                    x_left >= 0 && y_top >= 0 &&
                    x_right < frame_width && y_bottom < frame_height)
                {
                    boxes_vec.push_back(Rect(x_left, y_top, w, h));
                    ids_vec.push_back(class_id);
                }
            }
        }

        // 5. 输出结果
        Mat(boxes_vec).copyTo(out_boxes_mat);
        Mat(ids_vec).copyTo(out_ids_mat);

    } catch (const cv::Exception &e) {
        // 捕获 OpenCV 内部异常时输出 LOGE
        LOGE("OpenCV Error in detectObjectsNN: %s", e.what());
    } catch (...) {
        LOGE("Unknown Error in detectObjectsNN");
    }
}