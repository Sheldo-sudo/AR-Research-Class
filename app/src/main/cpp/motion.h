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