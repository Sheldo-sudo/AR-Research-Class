#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "enhance.h"  // 包含 MSRCR、CLAHE 相关处理函数
#include "videoStab.h"  // 视频稳定相关函数
#include "dehaze.h"   // 包含去雾相关处理函数

#define TAG "NativeLib"

using namespace std;
using namespace cv;

// MSRCR 增强方法
void enhanceByMSRCR(Mat& image) {
    MSRCR(image, 1.2, 146, 200);  // 调用 enhance.cpp 中的 MSRCR 函数
}

// CLAHE 增强方法
void enhanceByCLAHE(Mat& image) {
    Ptr<CLAHE> clahe = createCLAHE();
    setClaheParams(clahe, image);  // 调用 enhance.cpp 中的 setClaheParams 函数
    image = hsv_clahe(image, clahe);  // 调用 enhance.cpp 中的 hsv_clahe 函数
}

// 去雾方法
void dehaze(Mat& image) {
    dehazeProcess(image, image);  // 调用 dehaze.cpp 中的 dehazeProcess 函数
}

extern "C" {
// 视频稳像处理函数
JNIEXPORT void JNICALL
Java_com_example_glasspro_NativeProcessor_videoStab(JNIEnv *env, jclass clazz, jlong matAddr1,
                                                    jlong matAddr2) {
    // 获取图像 Mat 对象
    Mat &frame1 = *(Mat *) matAddr1;
    Mat &frame2 = *(Mat *) matAddr2;

    // 创建 VideoStab 对象
    VideoStab stab;

    // 获取加速度数据（假设传递了加速度数据，或者用默认值）
    float accelX = 0.0, accelY = 0.0, accelZ = 0.0;

    // 判断是否需要进行视频稳像
    bool isMoving = stab.shouldStabilize(accelX, accelY, accelZ);  // 通过 VideoStab 实例调用

    if (isMoving) {
        // 设备在运动，进行视频稳像
        frame2 = stab.stabilize(frame1, frame2, accelX, accelY, accelZ);
    } else {
        // 设备静止，根据图像亮度选择增强方法
        Scalar meanVal = mean(frame2);  // 计算图像的平均值（亮度）
        double brightness = meanVal[0];  // 亮度值

        if (brightness < 50) {
            // 亮度较低，使用 MSRCR 增强图像
            enhanceByMSRCR(frame2);
        } else if (brightness > 200) {
            // 亮度较高，使用 CLAHE 增强图像
            enhanceByCLAHE(frame2);
        } else {
            // 亮度适中，使用去雾方法
            dehaze(frame2);
        }
    }

    // 将处理后的图像传回
    // 可以通过 JNI 方式将处理后的图像传回 Java 层
}

// 空域去噪（双边滤波）
void applyBilateralFilter(Mat &image, double noise_level) {
    cv::Mat temp;
    cv::bilateralFilter(image, temp, 9, noise_level * 50, noise_level * 50);
    temp.copyTo(image);
}
}

// 自动图像处理（选择合适的图像增强或去雾方法）
extern "C" {
JNIEXPORT void JNICALL
Java_com_example_glasspro_NativeProcessor_autoProcess(JNIEnv *env, jclass clazz, jlong matAddr1,
                                                      jlong matAddr2, jfloat accelX, jfloat accelY,
                                                      jfloat accelZ) {
    // 获取图像 Mat 对象
    Mat &frame1 = *(Mat *) matAddr1;
    Mat &frame2 = *(Mat *) matAddr2;

    // 判断是否需要进行视频稳像
    VideoStab stab;
    bool isMoving = stab.shouldStabilize(accelX, accelY, accelZ); // 判断设备是否在运动

    if (isMoving) {
        // 设备在运动，进行视频稳像
        frame2 = stab.stabilize(frame1, frame2, accelX, accelY, accelZ);
    } else {
        // 设备静止，根据图像亮度选择增强方法
        Scalar meanVal = mean(frame2);  // 计算图像的平均值（亮度）
        double brightness = meanVal[0];  // 亮度值

        if (brightness < 50) {
            // 亮度较低，使用 MSRCR 增强图像
            enhanceByMSRCR(frame2);
        } else if (brightness > 200) {
            // 亮度较高，使用 CLAHE 增强图像
            enhanceByCLAHE(frame2);
        } else {
            // 亮度适中，使用去雾方法
            dehaze(frame2);
        }
    }

    // 将处理后的图像传回
    // 可以通过 JNI 方式将处理后的图像传回 Java 层
}
}
// 生成Retinex不同尺度的分布（如果需要使用此方法）
vector<double> retinexScalesDistribution(double maxScale, int nScales) {
    vector<double> scales;
    double scaleStep = maxScale / nScales;
    for (int i = 0; i < nScales; ++i) {
        scales.push_back(scaleStep * i + 2.0);
    }
    return scales;
}

// 其他的图像处理方法
Mat replaceZeroes(const Mat &data) {
    Mat mask = (data != 0);
    double min_nonzero;
    minMaxLoc(data, &min_nonzero, nullptr, nullptr, nullptr, mask);
    Mat output = data.clone();
    output.setTo(min_nonzero, ~mask);
    return output;
}

void separableGaussianBlur(const Mat &src, Mat &dst, double sigma) {
    int ksize = cvRound(sigma * 6 + 1);
    if (ksize % 2 == 0) ksize++;

    vector<float> kernel(ksize);
    int half = ksize / 2;
    float sum = 0.f;
    for (int i = 0; i < ksize; ++i) {
        int x = i - half;
        kernel[i] = expf(-0.5f * (x * x) / (sigma * sigma));
        sum += kernel[i];
    }
    for (int i = 0; i < ksize; ++i) kernel[i] /= sum;

    sepFilter2D(src, dst, -1, kernel, kernel);
}

Mat MSR(const Mat &img, const vector<double> &scales) {
    Mat logImg;
    log(img / 255.0, logImg);  // 归一化后取对数
    Mat log_R = Mat::zeros(img.size(), CV_32F);
    float weight = 1.0f / static_cast<float>(scales.size());

    for (double sigma: scales) {
        Mat blur, logBlur;
        separableGaussianBlur(img, blur, sigma);
        blur = replaceZeroes(blur);
        log(blur / 255.0, logBlur);
        log_R += weight * (logImg - logBlur);
    }
    return log_R;
}

// MSRCR 主函数
void MSRCR(Mat &inputImage, double dynamic = 4.0, double alpha = 20.0, double beta = 12.0) {
    Mat rgbImage;
    cvtColor(inputImage, rgbImage, COLOR_BGR2RGB);
    rgbImage.convertTo(rgbImage, CV_32FC3);  // 转换为 float

    vector<Mat> rgbChannels(3);
    split(rgbImage, rgbChannels);
    vector<double> scales = {10, 20};

    parallel_for_(Range(0, 3), [&](const Range &range) {
        for (int ch = range.start; ch < range.end; ++ch) {
            Mat &channel = rgbChannels[ch];
            channel = replaceZeroes(channel);

            Mat retinexChannel = MSR(channel, scales);

            // Retinex 输出归一化（拉伸对比度）
            Scalar meanVal, stdVal;
            meanStdDev(retinexChannel, meanVal, stdVal);

            double minVal = meanVal[0] - dynamic * stdVal[0];
            double maxVal = meanVal[0] + dynamic * stdVal[0];
            double range = std::max(maxVal - minVal, 1e-6);

            retinexChannel = (retinexChannel - minVal) * (255.0 / range);
            threshold(retinexChannel, retinexChannel, 255.0, 255.0, THRESH_TRUNC);
            threshold(retinexChannel, retinexChannel, 0.0, 0.0, THRESH_TOZERO);

            retinexChannel.convertTo(channel, CV_8U);  // 转换回 uint8
        }
    });

    merge(rgbChannels, rgbImage);
    cvtColor(rgbImage, inputImage, COLOR_RGB2BGR);  // 转回 BGR
}
