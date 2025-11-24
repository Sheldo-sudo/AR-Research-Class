#ifndef NATIVE_OPENCV_ANDROID_TEMPLATE_ENHANCE_H
#define NATIVE_OPENCV_ANDROID_TEMPLATE_ENHANCE_H
#include <cstdio>
#include <iostream>
#include <cstdlib>
#include <cmath>
#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// 定义参数

extern int levels;
extern TermCriteria termcrit;
extern Size winsize;

// 设置CLAHE参数
void setClaheParams(Ptr<CLAHE> clahePtr, const Mat& inputImage);

// 插值函数
Mat interpolation(const Mat& p_c, const Mat& dis, double noise_level = 1.5, double c_noise = 1.0, double c_mid = 1.0);

// 构建高斯金字塔
vector<Mat> buildGaussianPyramid(const Mat& image, int numLevels = 3);

// 构建拉普拉斯金字塔
vector<Mat> buildLaplacianPyramid(const vector<Mat>& gaussianPyramid, int levels = 3);

// 拉普拉斯金字塔融合
Mat laplacianPyramidFusion(const vector<Mat>& pyramid1, const vector<Mat>& pyramid2);

// 对HSV颜色空间中的V通道应用CLAHE
Mat hsv_clahe(Mat& frame, const cv::Ptr<cv::CLAHE>& clahe);

// 处理帧
Mat processFrame(Mat& frame1, Mat& frame2, double noise_level, int levels = 3);

void clahe(Mat& image);

//// JNI 函数
//extern "C" {
//void JNICALL
//Java_com_example_nativeopencvandroidtemplate_MainActivity_enhance(JNIEnv *env, jobject instance, jlong matAddr1, jlong matAddr2, jdouble noise_level);}
//
//extern "C" {
//void JNICALL
//Java_com_example_nativeopencvandroidtemplate_MainActivity_enhanceByCLAHE(JNIEnv *env, jobject instance, jlong matAddr);}

//---------------------- msrcr ----------------------//
vector<double> retinexScalesDistribution(double maxScale, int nScales);
Mat rgbToHsv(const Mat& rgbImage);
Mat hsvToRgb(const Mat& hsvImage);
void MSRCR(Mat& inputImage, double dynamic,double alpha,double beta);
Mat replaceZeroes(const Mat& data);

#endif //NATIVE_OPENCV_ANDROID_TEMPLATE_ENHANCE_H
