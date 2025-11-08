#ifndef NATIVE_OPENCV_ANDROID_TEMPLATE_DEHAZE_H
#define NATIVE_OPENCV_ANDROID_TEMPLATE_DEHAZE_H
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


using namespace std;
using namespace cv;


//---------------------- Guided Filter -------------------//

//---------------------- Dehaze Functions -------------------//
double air_r, air_g, air_b;
Mat t;
int frameCnt;

// 在extract_blocks中将块加入堆时使用的Compare结构体
struct BlockCompare {
    bool operator()(const tuple<int, int, double>& a, const tuple<int, int, double>& b) {
        return get<2>(a) < get<2>(b); // 按 value 降序排列
    }
};

Mat laplacian_pyramid_fusion(Mat img1, Mat img2);
Mat GF_smooth(Mat& src, int s, double epsilon,int samplingRate);
Mat staticMin(Mat& I, int s, double eeps, double alpha, int samplingRate);
void est_air(Mat& R, Mat& G, Mat& B, int s, double* A_r, double* A_g, double* A_b);
void est_air_patchwise(const Mat& R, const Mat& G, const Mat& B,
                       int patchSize, Mat& A_R, Mat& A_G, Mat& A_B);
Mat est_trans_fast(Mat& R, Mat& G, Mat& B, int s, double eeps, double k, Mat& A_r, Mat& A_g, Mat& A_b);
//Mat est_trans_fast(Mat& R, Mat& G, Mat& B, int s, double eeps, double k, double A_r, double A_g, double A_b);
Mat rmv_haze(Mat& R, Mat& G, Mat& B, Mat& t, Mat& A_r, Mat& A_g, Mat& A_b);
//Mat rmv_haze(Mat& R, Mat& G, Mat& B, Mat& t, double A_r, double A_g, double A_b);
//void post_process(vector<Mat>& channels, double thresholdPercentage);
void dehazeProcess(Mat& image, Mat& dehazedImage);
Vec3f computePatchLine(const Mat& patch);
vector<Vec3f> selectPatches(const Mat& image, int patchSize);
Vec3f estimateAtmosphericLightDirection(const vector<Vec3f>& lines);
float estimateAtmosphericLightMagnitude(const Mat& image, const Vec3f& direction);
void airlight_automatic_estimate(Mat& hazyImage);
Mat extract_blocks(const Mat& input_img, const Mat& guide_img, int patchSize);

extern "C" {
void JNICALL
Java_com_example_nativeopencvandroidtemplate_MainActivity_dehaze(JNIEnv *env, jclass clazz, jlong matAddr);}

#endif //NATIVE_OPENCV_ANDROID_TEMPLATE_DEHAZE_H
