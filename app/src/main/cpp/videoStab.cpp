#include "videoStab.h"
#include <cmath>
#include <opencv2/opencv.hpp>
#include <jni.h>

using namespace cv;
using namespace std;

// Parameters for Kalman Filter
#define Q1 0.000001
#define R1 0.00001

// 调试模式
#define test 0

// [!! 修复 3：添加缺失的定义 !!]
// 这个值在你的代码中被使用了 (line 73, 186)，但是从未被定义。
// 它代表从左右边缘裁剪的像素数，以隐藏稳像带来的黑边。
// 30 是一个合理的默认值。
#define HORIZONTAL_BORDER_CROP 30

int height;
int width;

VideoStab::VideoStab()
{
    smoothedMat.create(2, 3, CV_64F);

    k = 1;

    errscaleX = 1;
    errscaleY = 1;
    errthetha = 1;
    errtransX = 1;
    errtransY = 1;

    Q_scaleX = Q1;
    Q_scaleY = Q1;
    Q_thetha = Q1;
    Q_transX = Q1;
    Q_transY = Q1;

    R_scaleX = R1;
    R_scaleY = R1;
    R_thetha = R1;
    R_transX = R1;
    R_transY = R1;

    sum_scaleX = 0;
    sum_scaleY = 0;
    sum_thetha = 0;
    sum_transX = 0;
    sum_transY = 0;

    scaleX = 0;
    scaleY = 0;
    thetha = 0;
    transX = 0;
    transY = 0;

    // 初始化加速度相关参数
    accel_threshold_min = 0.3;  // 低于此值认为设备静止
    accel_threshold_max = 3.0;  // 高于此值认为是剧烈运动
    last_accel_x = 0;
    last_accel_y = 0;
    last_accel_z = 0;
}

// 计算加速度矢量大小
double VideoStab::calculateAccelMagnitude(double x, double y, double z)
{
    return sqrt(x*x + y*y + z*z);
}

// 判断是否需要稳像
bool VideoStab::shouldStabilize(double accel_x, double accel_y, double accel_z)
{
    // 计算当前加速度矢量大小
    double current_accel = calculateAccelMagnitude(accel_x, accel_y, accel_z);

    // 计算加速度变化率
    double delta_accel = abs(current_accel - calculateAccelMagnitude(last_accel_x, last_accel_y, last_accel_z));

    // 更新上一次加速度值
    last_accel_x = accel_x;
    last_accel_y = accel_y;
    last_accel_z = accel_z;

    // 判断条件
    if (current_accel < accel_threshold_min) {
        // 加速度太小，设备基本静止
        return false;
    } else if (current_accel > accel_threshold_max) {
        // 加速度太大，可能是剧烈运动或碰撞
        return false;
    } else if (delta_accel > 2.0) {
        // 加速度变化太大，可能是突然运动
        return false;
    }

    return true;
}

Mat VideoStab::stabilize(Mat frame_1, Mat frame_2, double accel_x, double accel_y, double accel_z)
{
    // 首先检查加速度条件
//    if (!shouldStabilize(accel_x, accel_y, accel_z)) {
//        return frame_1.clone();
//    }

    cvtColor(frame_1, frame1, COLOR_BGR2GRAY);
    cvtColor(frame_2, frame2, COLOR_BGR2GRAY);

    int vert_border = HORIZONTAL_BORDER_CROP * frame_1.rows / frame_1.cols;

    vector<Point2f> features1, features2;
    vector<Point2f> goodFeatures1, goodFeatures2;
    vector<uchar> status;
    vector<float> err;

    goodFeaturesToTrack(frame1, features1, 200, 0.01, 30);
    calcOpticalFlowPyrLK(frame1, frame2, features1, features2, status, err);

    for (size_t i = 0; i < status.size(); i++)
    {
        if (status[i])
        {
            goodFeatures1.push_back(features1[i]);
            goodFeatures2.push_back(features2[i]);
        }
    }

    if (goodFeatures1.size() < 10 || goodFeatures2.size() < 10) {
        // 特征点太少，返回原图
        return frame_1.clone();
    }

    affine = estimateAffinePartial2D(goodFeatures1, goodFeatures2);

    if (affine.empty())
    {
        return frame_1.clone();
    }

    dx = affine.at<double>(0, 2);
    dy = affine.at<double>(1, 2);
    da = atan2(affine.at<double>(1, 0), affine.at<double>(0, 0));
    ds_x = affine.at<double>(0, 0) / cos(da);
    ds_y = affine.at<double>(1, 1) / cos(da);

    sx = ds_x;
    sy = ds_y;

    sum_transX += dx;
    sum_transY += dy;
    sum_thetha += da;
    sum_scaleX += ds_x;
    sum_scaleY += ds_y;

    if (k == 1)
    {
        k++;
    }
    else
    {
        Kalman_Filter(&scaleX, &scaleY, &thetha, &transX, &transY);
    }

    diff_scaleX = scaleX - sum_scaleX;
    diff_scaleY = scaleY - sum_scaleY;
    diff_transX = transX - sum_transX;
    diff_transY = transY - sum_transY;
    diff_thetha = thetha - sum_thetha;

    ds_x = ds_x + diff_scaleX;
    ds_y = ds_y + diff_scaleY;
    dx = dx + diff_transX;
    dy = dy + diff_transY;
    da = da + diff_thetha;

    smoothedMat.at<double>(0, 0) = sx * cos(da);
    smoothedMat.at<double>(0, 1) = sx * -sin(da);
    smoothedMat.at<double>(1, 0) = sy * sin(da);
    smoothedMat.at<double>(1, 1) = sy * cos(da);

    smoothedMat.at<double>(0, 2) = dx;
    smoothedMat.at<double>(1, 2) = dy;

    warpAffine(frame_1, smoothedFrame, smoothedMat, frame_2.size());

    // [!! 修复 4：修复拼写错误 !!]
    // 将 (小写) horizontal_border_crop 修正为 (大写) HORIZONTAL_BORDER_CROP
    smoothedFrame = smoothedFrame(Range(vert_border, smoothedFrame.rows - vert_border),
                                  Range(HORIZONTAL_BORDER_CROP, smoothedFrame.cols - HORIZONTAL_BORDER_CROP));

    resize(smoothedFrame, smoothedFrame, frame_2.size());

    // test关闭，不做imshow
    return smoothedFrame;
}

void VideoStab::Kalman_Filter(double* scaleX, double* scaleY, double* thetha, double* transX, double* transY)
{
    double frame_1_scaleX = *scaleX;
    double frame_1_scaleY = *scaleY;
    double frame_1_thetha = *thetha;
    double frame_1_transX = *transX;
    double frame_1_transY = *transY;

    double frame_1_errscaleX = errscaleX + Q_scaleX;
    double frame_1_errscaleY = errscaleY + Q_scaleY;
    double frame_1_errthetha = errthetha + Q_thetha;
    double frame_1_errtransX = errtransX + Q_transX;
    double frame_1_errtransY = errtransY + Q_transY;

    double gain_scaleX = frame_1_errscaleX / (frame_1_errscaleX + R_scaleX);
    double gain_scaleY = frame_1_errscaleY / (frame_1_errscaleY + R_scaleY);
    double gain_thetha = frame_1_errthetha / (frame_1_errthetha + R_thetha);
    double gain_transX = frame_1_errtransX / (frame_1_errtransX + R_transX);
    double gain_transY = frame_1_errtransY / (frame_1_errtransY + R_transY);

    *scaleX = frame_1_scaleX + gain_scaleX * (sum_scaleX - frame_1_scaleX);
    *scaleY = frame_1_scaleY + gain_scaleY * (sum_scaleY - frame_1_scaleY);
    *thetha = frame_1_thetha + gain_thetha * (sum_thetha - frame_1_thetha);
    *transX = frame_1_transX + gain_transX * (sum_transX - frame_1_transX);
    *transY = frame_1_transY + gain_transY * (sum_transY - frame_1_transY);

    errscaleX = (1 - gain_scaleX) * frame_1_errscaleX;
    errscaleY = (1 - gain_scaleY) * frame_1_errscaleY;
    errthetha = (1 - gain_thetha) * frame_1_errthetha;
    errtransX = (1 - gain_transX) * frame_1_errtransX;
    errtransY = (1 - gain_transY) * frame_1_errtransY;
}

// JNI 函数 (来自回答 #14，保持不变，它已经是正确的)
extern "C" {
void JNICALL
Java_com_example_glasspro_NativeProcessor_videoStab(JNIEnv *env, jclass clazz, jlong mat_addr1,
                                                    jlong mat_addr2) {
    // 从原始地址获取 Mat 对象
    Mat frame1 = *(Mat *) mat_addr1;
    Mat frame2 = *(Mat *) mat_addr2;

    Mat &image = *(Mat *) mat_addr2;
    width = image.cols;
    height = image.rows;

    // [!! 修复 1：致命Bug !!]
    // 将 stab 对象声明为 static。
    static VideoStab stab;

    // [!! 修复 2：编译错误 !!]
    // 传入 0.0 作为虚拟的加速度值
    image = stab.stabilize(frame1, frame2, 0.0, 0.0, 0.0);
}
}