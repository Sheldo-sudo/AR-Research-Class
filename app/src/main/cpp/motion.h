#ifndef GLASSPRO_MOTION_H
#define GLASSPRO_MOTION_H

#include <opencv2/opencv.hpp>
#include <vector> // 包含 vector

/**
 * @brief 检测运动物体，并将轮廓的边界框存入 out_boxes。
 * @param prev_frame_rgba 上一帧的 RGBA Mat
 * @param current_frame_rgba 当前帧的 RGBA Mat
 * @param out_boxes [输出参数] 用于存储检测到的 cv::Rect 的向量
 */
void detectMotion(const cv::Mat& prev_frame_rgba,
                  const cv::Mat& current_frame_rgba,
                  std::vector<cv::Rect>& out_boxes); // <-- [!! 签名大修改 !!]

#endif //GLASSPRO_MOTION_H
/*
建议：有意义无意义的运动（无意义指随机的抖动之类的）使用相关方法区分出来，在有意义的运动中，相机不动的话比较简单，相机如果也运动，那么可以调用相机中的模块（？）来去掉这一部分
目标检测，检测人脸、汽车等就行，做一些比较大的人脸，可使用轻量网络，看看相机有没有相关模块
我的工作（一头一尾）：最开始视频处理部分的稳像，最后是增加目标检测
 */