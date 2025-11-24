#ifndef NATIVE_OPENCV_ANDROID_TEMPLATE_STAB_H
#define NATIVE_OPENCV_ANDROID_TEMPLATE_STAB_H

#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <tuple>
#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;
using namespace std;


class VideoStab
{
public:
    VideoStab();
    VideoCapture capture;

    Mat frame2;
    Mat frame1;

    int k;

    const int HORIZONTAL_BORDER_CROP = 30;

    Mat smoothedMat;
    Mat affine;

    Mat smoothedFrame;

    double dx ;
    double dy ;
    double da ;
    double ds_x ;
    double ds_y ;

    double sx ;
    double sy ;

    double scaleX ;
    double scaleY ;
    double thetha ;
    double transX ;
    double transY ;

    double diff_scaleX ;
    double diff_scaleY ;
    double diff_transX ;
    double diff_transY ;
    double diff_thetha ;

    double errscaleX ;
    double errscaleY ;
    double errthetha ;
    double errtransX ;
    double errtransY ;

    double Q_scaleX ;
    double Q_scaleY ;
    double Q_thetha ;
    double Q_transX ;
    double Q_transY ;

    double R_scaleX ;
    double R_scaleY ;
    double R_thetha ;
    double R_transX ;
    double R_transY ;

    double sum_scaleX ;
    double sum_scaleY ;
    double sum_thetha ;
    double sum_transX ;
    double sum_transY ;

    // 新增加速度相关参数
    double accel_threshold_min;  // 最小加速度阈值
    double accel_threshold_max;  // 最大加速度阈值
    double last_accel_x;         // 上一次x轴加速度
    double last_accel_y;         // 上一次y轴加速度
    double last_accel_z;         // 上一次z轴加速度

    Mat stabilize(Mat frame_1 , Mat frame_2 , double accel_x = 0 , double accel_y = 0 , double accel_z = 0);
    void Kalman_Filter(double *scaleX , double *scaleY , double *thetha , double *transX , double *transY);
    bool shouldStabilize(double accel_x, double accel_y, double accel_z);
private:
    double calculateAccelMagnitude(double x, double y, double z);
};



//extern "C" {
//void JNICALL
//Java_com_example_nativeopencvandroidtemplate_MainActivity_videoStab(JNIEnv *env, jobject instance, jlong matAddr1, jlong matAddr2);}

#endif // NATIVE_OPENCV_ANDROID_TEMPLATE_STAB_H