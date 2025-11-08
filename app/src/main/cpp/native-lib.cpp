#include <jni.h>
#include <android/log.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include "enhance.h"
#include "videoStab.h"
#include "dehaze.h"

#define TAG "NativeLib"

using namespace std;
using namespace cv;

extern "C" {
void JNICALL
Java_com_example_nativeopencvandroidtemplate_MainActivity_adaptiveThresholdFromJNI(JNIEnv *env,
                                                                                   jobject instance,
                                                                                   jlong matAddr) {

    // get Mat from raw address
    Mat &mat = *(Mat *) matAddr;

    clock_t begin = clock();

    cv::adaptiveThreshold(mat, mat, 255, ADAPTIVE_THRESH_MEAN_C, THRESH_BINARY_INV, 21, 5);

    // log computation time to Android Logcat
    double totalTime = double(clock() - begin) / CLOCKS_PER_SEC;
    __android_log_print(ANDROID_LOG_INFO, TAG, "adaptiveThreshold computation time = %f seconds\n",
                        totalTime);
}
}

extern "C"
JNIEXPORT void JNICALL
Java_com_example_glasspro_NativeProcessor_autoProcess(JNIEnv *env, jclass, jlong matAddr) {
	Mat &img = *(Mat *) matAddr;
	if (img.empty()) return;

	// 缩小图像，假设原图很大，缩放到320x240，降低计算量
	Mat resized;
	resize(img, resized, Size(320, 240));

	// 转HSV，避免多次转色
	Mat hsv;
	cvtColor(resized, hsv, COLOR_RGBA2RGB);  // 先转RGB
	cvtColor(hsv, hsv, COLOR_RGB2HSV);

	// 分离HSV通道
	std::vector<Mat> channels;
	split(hsv, channels);
	Mat& H = channels[0];
	Mat& S = channels[1];
	Mat& V = channels[2];  // V通道代替亮度

	// 计算亮度均值和标准差（用V通道）
	Scalar meanVal, stddevVal;
	meanStdDev(V, meanVal, stddevVal);
	double brightness = meanVal[0];
	double contrast = stddevVal[0];

	// 饱和度均值
	Scalar satMean = mean(S);

	Mat gray;
	cvtColor(resized, gray, COLOR_RGBA2GRAY);
	Mat lap;
	Laplacian(gray, lap, CV_64F);
	Scalar lapMean, lapStddev;
	meanStdDev(lap, lapMean, lapStddev);
	double blurMeasure = lapStddev[0] * lapStddev[0];
	if (blurMeasure<100) {
		// 调用稳像函数

	}

	if (brightness < 50) {
		// 低亮度，调用MSRCR增强函数
		// MSRCR(img);
	}else if (brightness > 200) {
		// 高亮度，调用CLAHE函数
		// CLAHE(img);
	}

	if (contrast < 30 && satMean[0] < 50) {
		// 低对比度且饱和度低，可能是雾霾，调用去雾函数
		// Dehaze(img);
	}
}

