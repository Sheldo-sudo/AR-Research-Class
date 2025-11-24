//
// Created by 罗孝俊 on 2024/12/8.
//

#ifndef VIDEO_ENHANCE_THRESHOLD_H
#define VIDEO_ENHANCE_THRESHOLD_H

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/opencv.hpp>

using namespace cv;

Mat threshold_image_output(const Mat &input_img, int patchSize);
#endif //VIDEO_ENHANCE_THRESHOLD_H
