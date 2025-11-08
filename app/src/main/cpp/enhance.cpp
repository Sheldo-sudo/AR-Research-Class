#include "enhance.h"
#include <opencv2/opencv.hpp>
// Parameters for pyraimids
int levels = 3;
int height;
int width;
// Parameters for Lucas Kanade Optical Flow
TermCriteria termcrit(TermCriteria::COUNT | TermCriteria::EPS, 10, 0.03);
Size winsize = Size(16, 16);

// Parameters for CLAHE
void setClaheParams(Ptr<CLAHE> clahePtr, const Mat& inputImage) {
    Mat image_gray;
    cv::cvtColor(inputImage, image_gray, COLOR_BGR2GRAY);
    double mean = cv::mean(image_gray)[0];
    mean = 0.5 - mean / 128;
    clahePtr->setClipLimit(20 / (1 + cv::exp(-mean * 20)));
    clahePtr->setTilesGridSize(Size(4, 4));
}



vector<Mat> buildGaussianPyramid(const Mat& image, int numLevels) {
    vector<Mat> gaussianPyramid;
    gaussianPyramid.push_back(image);
    for (int i = 1; i < numLevels; ++i) {
        Mat downsampled;
        pyrDown(gaussianPyramid.back(), downsampled);
        gaussianPyramid.push_back(downsampled);
    }
    return gaussianPyramid;
}

vector<Mat> buildLaplacianPyramid(const vector<Mat>& gaussianPyramid, int levels) {
    vector<Mat> laplacianPyramid;
    for (int i = 0; i < levels - 1; ++i) {
        Mat upsampled;
        pyrUp(gaussianPyramid[i + 1], upsampled, gaussianPyramid[i].size());
        Mat laplacian;
        subtract(gaussianPyramid[i], upsampled, laplacian);
        laplacianPyramid.push_back(laplacian);
    }
    laplacianPyramid.push_back(gaussianPyramid[levels - 1]);
    for (int i = 0; i < levels; i++){
        laplacianPyramid[i].convertTo(laplacianPyramid[i], CV_32FC3, 1 / 255.0);
    }
    return laplacianPyramid;
}

Mat interpolation(const Mat& p_c, const Mat& dis, double noise_level, double c_noise, double c_mid) {
    double m;
    Mat temp;
    m = 1 + c_mid * (1 - cv::exp(-noise_level * c_noise));
    cv::exp(-(abs(dis) - m), temp);
    Mat i_pyramid = 1 / (1 + temp);

    return i_pyramid;
}

Mat laplacianPyramidFusion(const vector<Mat>& pyramid1, const vector<Mat>& pyramid2) {
    vector<Mat> fusedPyramid;
    Mat p1, p2, dis, fused;
    for (size_t i = 0; i < levels; ++i) {
        p1 = pyramid1[i];
        p2 = pyramid2[i];
        absdiff(p1, p2, dis);
        Mat itp = interpolation(p2, dis);
        p2 += itp.mul(dis);
        addWeighted(p1, 0.9, p2, 0.1, 0, fused);
        fusedPyramid.push_back(fused);
    }

    Mat fusedFrame = fusedPyramid[fusedPyramid.size() - 1];
    for (int i = static_cast<int>(fusedPyramid.size()) - 1; i > 0; --i) {
        pyrUp(fusedFrame, fusedFrame, fusedPyramid[i - 1].size());
        add(fusedFrame, fusedPyramid[i - 1], fusedFrame);
    }
    Mat result;
    fusedFrame.convertTo(result, CV_8U, 255.0);
    return result;
}

Mat hsv_clahe(Mat& frame, const cv::Ptr<cv::CLAHE>& clahe) {
    // 将BGR图像转换为HSV颜色空间
    Mat hsv_frame;
    cvtColor(frame, hsv_frame, COLOR_BGR2HSV);
    // 分离HSV通道
    vector<Mat> channels;
    split(hsv_frame, channels);
    // 对V通道应用CLAHE
    clahe->apply(channels[2], channels[2]);
    // 合并HSV通道
    merge(channels, hsv_frame);
    // 将HSV图像转换回BGR颜色空间
    Mat result;
    cvtColor(hsv_frame, result, COLOR_HSV2BGR);
    return result;
}

Mat processFrame(Mat& frame1, Mat& frame2, double noise_level, int levels) {

    double rate = 1;
    resize(frame1, frame1, Size(frame1.cols / rate, frame1.rows / rate));
    resize(frame2, frame2, Size(frame2.cols / rate, frame2.rows / rate));

    cvtColor(frame1, frame1, COLOR_RGBA2RGB);
    cvtColor(frame2, frame2, COLOR_RGBA2RGB);

    Ptr<CLAHE> clahe = createCLAHE();
    setClaheParams(clahe, frame2);
    frame2 = hsv_clahe(frame2, clahe);

    vector<Mat> pyramidPrev = buildGaussianPyramid(frame1, levels);
    vector<Mat> lapPrev = buildLaplacianPyramid(pyramidPrev);

    vector<Mat> pyramidCurrent = buildGaussianPyramid(frame2, levels);
    vector<Mat> lapCurrent = buildLaplacianPyramid(pyramidCurrent);

    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "Start enhancing1");
    // 在每一层上应用光流
    for (int i = levels - 1; i >= 0; --i) {
        cv::Mat src, target;
        cvtColor(pyramidPrev[i], src, COLOR_BGR2GRAY);
        cvtColor(pyramidCurrent[i], target, COLOR_BGR2GRAY);
        std::vector<cv::Point2f> src_points, target_points;
        cv::goodFeaturesToTrack(src, src_points, 100, 0.3, 7, cv::Mat());

        std::vector<uchar> status;
        std::vector<float> err;

        if (src_points.empty()) {
            // 安全退出或返回原图，防止崩溃
            resize(frame2, frame2, Size(width, height));
            return frame2;
        }

        calcOpticalFlowPyrLK(src, target, src_points, target_points, status, err, winsize, 3, termcrit, 0);

        // 过滤掉未成功追踪的点
        std::vector<cv::Point2f> good_src_points, good_target_points;
        for (size_t i = 0; i < src_points.size(); i++) {
            if (status[i]) {
                good_src_points.push_back(src_points[i]);
                good_target_points.push_back(target_points[i]);
            }
        }
        if (good_target_points.size()<4){
            resize(frame2, frame2, Size(width, height));
            return frame2;
        }

        // 计算变换矩阵
        cv::Mat H = cv::findHomography(good_src_points, good_target_points, cv::RANSAC);
        Mat aligned;

        if (H.empty()) {
            __android_log_print(ANDROID_LOG_ERROR, "OpenCV", "findHomography failed, H is empty at level %d", i);
            resize(frame2, frame2, Size(width, height));
            return frame2;
        }

        // 应用变换矩阵
        cv::warpPerspective(lapPrev[i], aligned, H, lapPrev[i].size());
        lapPrev[i] = aligned;
    }

    Mat resFrame = laplacianPyramidFusion(lapCurrent, lapPrev);

    resize(resFrame, resFrame, Size(width, height));
    frame1.release();
    frame2.release();
    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "Ending");
    return resFrame;
}


// JNI 函数
extern "C" {
void JNICALL
Java_com_example_glasspro_NativeProcessor_enhance(JNIEnv *env,jclass clazz, jlong matAddr1, jlong matAddr2, jdouble noise_level) {

// 从原始地址获取 Mat 对象
Mat frame1 = *(Mat *) matAddr1;
Mat frame2 = *(Mat *) matAddr2;

Mat &image = *(Mat *) matAddr2;
width = image.cols;
height = image.rows;

// 调用 processFrame 算法
image = processFrame(frame1, frame2, noise_level);
// 空域去噪（双边滤波）
cv::Mat temp;
cv::bilateralFilter(image, temp,
                        9,
                        noise_level * 50,
                        noise_level * 50);

// 将结果拷贝回原始图像
temp.copyTo(image);
}
}
void clahe(Mat& image) {
	if (image.empty()) return;

	cvtColor(image, image, COLOR_RGBA2RGB);
	Ptr<CLAHE> clahe = createCLAHE();
	setClaheParams(clahe, image);
	image = hsv_clahe(image, clahe);
}

//void clahe(Mat& image) {
//    if (image.empty()) return;
//
//    // --- 新增代码开始 ---
//    // 在所有处理前，先进行一次快速的色彩降噪。
//    cv::fastNlMeansDenoisingColored(image, image, 3, 3, 7, 21);
//    // --- 新增代码结束 ---
//
//    cvtColor(image, image, COLOR_RGBA2RGB);
//    Ptr<CLAHE> clahe = createCLAHE();
//    setClaheParams(clahe, image);
//    image = hsv_clahe(image, clahe);
//}

extern "C" {
void JNICALL
Java_com_example_glasspro_NativeProcessor_enhanceByCLAHE
		(JNIEnv *env, jclass clazz, jlong matAddr) {

	Mat &image = *(Mat *) matAddr;
	clahe(image);
}
}


//extern "C" {
//void JNICALL
//Java_com_example_glasspro_NativeProcessor_enhanceByCLAHE
//        (JNIEnv *env, jclass clazz, jlong matAddr) {
//    Mat &image = *(Mat *) matAddr;
//
//    // --- 新增代码开始 ---
//    // 快速检查亮度，如果画面足够亮 (例如平均亮度 > 100)，就没必要增强
//    Mat temp_gray;
//    // 注意：这里用RGBA2GRAY，因为从Java层传来的原始图像是RGBA格式
//    cv::cvtColor(image, temp_gray, COLOR_RGBA2GRAY);
//    if (cv::mean(temp_gray)[0] > 100) {
//        return;
//    }
//    // --- 新增代码结束 ---
//
//    clahe(image);
//}
//}
//// 生成 Retinex 不同尺度的分布
//vector<double> retinexScalesDistribution(double maxScale, int nScales) {
//    vector<double> scales;
//    double scaleStep = maxScale / nScales;
//    for (int i = 0; i < nScales; ++i) {
//        scales.push_back(scaleStep * i + 2.0);
//    }
//    return scales;
//}


Mat replaceZeroes(const Mat& data) {
	Mat mask = (data != 0);
	double min_nonzero;
	minMaxLoc(data, &min_nonzero, nullptr, nullptr, nullptr, mask);
	Mat output = data.clone();
	output.setTo(min_nonzero, ~mask);
	return output;
}

void separableGaussianBlur(const Mat& src, Mat& dst, double sigma) {
	int ksize = cvRound(sigma * 6 + 1);
	if (ksize % 2 == 0) ksize++;

	// 构造一维高斯核
	vector<float> kernel(ksize);
	int half = ksize / 2;
	float sum = 0.f;
	for (int i = 0; i < ksize; ++i) {
		int x = i - half;
		kernel[i] = expf(-0.5f * (x * x) / (sigma * sigma));
		sum += kernel[i];
	}
	for (int i = 0; i < ksize; ++i) kernel[i] /= sum;

	// 使用 OpenCV 内建的分离卷积
	sepFilter2D(src, dst, -1, kernel, kernel);
}

// 多尺度 Retinex
Mat MSR(const Mat& img, const vector<double>& scales) {
	Mat logImg;
	log(img / 255.0, logImg);  // 归一化后取对数
	Mat log_R = Mat::zeros(img.size(), CV_32F);
	float weight = 1.0f / static_cast<float>(scales.size());

	for (double sigma : scales) {
		Mat blur, logBlur;
		separableGaussianBlur(img, blur, sigma);
		blur = replaceZeroes(blur);
		log(blur / 255.0, logBlur);
		log_R += weight * (logImg - logBlur);
	}
	return log_R;
}

// MSRCR 主函数
void MSRCR(Mat& inputImage, double dynamic = 4.0, double alpha = 20.0, double beta = 12.0) {
	Mat rgbImage;
	cvtColor(inputImage, rgbImage, COLOR_BGR2RGB);
	rgbImage.convertTo(rgbImage, CV_32FC3);  // 统一转 float

	vector<Mat> rgbChannels(3);
	split(rgbImage, rgbChannels);
	vector<double> scales = {10, 20};

	parallel_for_(Range(0, 3), [&](const Range& range) {
		for (int ch = range.start; ch < range.end; ++ch) {
			Mat& channel = rgbChannels[ch];
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

extern "C" {
void JNICALL
Java_com_example_glasspro_NativeProcessor_enhanceByMSRCR(JNIEnv *env, jclass clazz, jlong matAddr) {

	// 获取传入的Mat图像
	Mat &image = *(Mat *)matAddr;  // 通过类型转换获取图像引用
	// 调用 MSRCR 函数，直接修改传入的 image
	MSRCR(image,1.2,146,200);

	//	Ptr<CLAHE> clahe = createCLAHE();
	//	setClaheParams(clahe, image);
	//	image = hsv_clahe(image, clahe);
}
}

