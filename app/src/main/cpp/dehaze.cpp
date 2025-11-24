#include "dehaze.h"
#include "threshold.h"


int width;
int height;

extern "C"
JNIEXPORT void JNICALL
Java_com_example_glasspro_NativeProcessor_dehaze(JNIEnv *env, jclass clazz, jlong matAddr) {
    // get Mat from raw address
    Mat &dehazedMat = *(Mat *) matAddr;
    Mat mat = *(Mat *) matAddr;
    // The image data which transmitted by Camera is RGBA, so we need to convert it to RGB
    cvtColor(mat, mat, COLOR_RGBA2RGB);

    width = dehazedMat.cols;
    height = dehazedMat.rows;
    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "Start dehazing. Mat rows: %d, cols: %d", dehazedMat.rows, dehazedMat.cols);

    dehazeProcess(mat, dehazedMat);

    dehazedMat.convertTo(dehazedMat, CV_8UC3);
    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "Converting image to CV_8UC3");
//    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "dehazedMat rows: %d, cols: %d", dehazedMat.rows, dehazedMat.cols);


//    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "Mat address: %p", matAddr);

}

// True method for processing
void dehazeProcess(Mat& image, Mat& dehazedImage) {
    frameCnt++;
    int s = 16;
    double rate = 1.2;
    double eeps = 0.002, omega = 0.9;

    resize(image, image, Size(image.cols / rate, image.rows / rate));
    image.convertTo(image, CV_64FC3);

    vector<Mat> channels(3);
    split(image, channels);
    Mat R = channels[2], G = channels[1], B = channels[0];

    Mat A_R20, A_G20, A_B20, t20, dehazed_20;
    Mat A_R80, A_G80, A_B80, t80, dehazed_80;

    // patchSize = 20
    est_air_patchwise(R, G, B, 20, A_R20, A_G20, A_B20);
    t20 = est_trans_fast(R, G, B, s, eeps, omega, A_R20, A_G20, A_B20);
    dehazed_20 = rmv_haze(R, G, B, t20, A_R20, A_G20, A_B20);

    // patchSize = 80
    est_air_patchwise(R, G, B, 80, A_R80, A_G80, A_B80);
    t80 = est_trans_fast(R, G, B, s, eeps, omega, A_R80, A_G80, A_B80);
    dehazed_80 = rmv_haze(R, G, B, t80, A_R80, A_G80, A_B80);

    // 多尺度融合
    dehazedImage = laplacian_pyramid_fusion(dehazed_20, dehazed_80);

    resize(dehazedImage, dehazedImage, Size(width, height));
    image.release();
}

// 拉普拉斯金字塔融合
Mat laplacian_pyramid_fusion(Mat img1, Mat img2) {
    int levels = 4;
    vector<Mat> gp1, gp2, lp1, lp2, lp_fused;
    Mat temp1 = img1.clone();
    Mat temp2 = img2.clone();

    // 构建高斯金字塔
    gp1.push_back(temp1);
    gp2.push_back(temp2);
    for (int i = 0; i < levels; ++i) {
        pyrDown(temp1, temp1);
        pyrDown(temp2, temp2);
        gp1.push_back(temp1);
        gp2.push_back(temp2);
    }

    // 构建拉普拉斯金字塔
    for (int i = 0; i < levels; ++i) {
        Mat up1, up2, lap1, lap2;
        pyrUp(gp1[i+1], up1, gp1[i].size());
        pyrUp(gp2[i+1], up2, gp2[i].size());
        lap1 = gp1[i] - up1;
        lap2 = gp2[i] - up2;
        lp1.push_back(lap1);
        lp2.push_back(lap2);
    }
    lp1.push_back(gp1[levels]);
    lp2.push_back(gp2[levels]);

    // 融合拉普拉斯金字塔
    for (int i = 0; i <= levels; ++i) {
        Mat fused = 0.5 * lp1[i] + 0.5 * lp2[i];
        lp_fused.push_back(fused);
    }

    // 重建图像
    Mat result = lp_fused[levels];
    for (int i = levels - 1; i >= 0; --i) {
        pyrUp(result, result, lp_fused[i].size());
        result += lp_fused[i];
    }
    return result;
}

//---------------------- GUIDED FILTER -------------------//
Mat GF_smooth(Mat& src, int s, double epsilon, int samplingRate)
{
    Mat srcResize;
//    src.convertTo(src, CV_64FC1);

    resize(src, srcResize, Size(src.cols / samplingRate, src.rows / samplingRate));
    s /= samplingRate;

    /*srcResize.convertTo(srcResize, CV_64FC1);*/
    Mat mean_I;
    blur(srcResize, mean_I, Size(s, s), Point(-1, -1));

    Mat II = srcResize.mul(srcResize);
    Mat var_I;
    blur(II, var_I, Size(s, s), Point(-1, -1));
    var_I = var_I - mean_I.mul(mean_I);

    Mat a = var_I / ((var_I + epsilon));
    Mat b = mean_I - a.mul(mean_I);

    blur(a, a, Size(s, s), Point(-1, -1));
    blur(b, b, Size(s, s), Point(-1, -1));
    resize(a, a, Size(src.cols, src.rows));
    resize(b, b, Size(src.cols, src.rows));

    return a.mul(src) + b;
}


Mat staticMin(Mat& I, int s, double eps, double alpha, int samplingRate)
{
    Mat mean_I = GF_smooth(I, s, eps, samplingRate);

    Mat var_I;
    blur((I - mean_I).mul(I - mean_I), var_I, Size(s, s), Point(-1, -1));

    Mat mean_var_I;
    blur(var_I, mean_var_I, Size(s, s), Point(-1, -1));

    Mat z_I;
    sqrt(mean_var_I, z_I);

    return mean_I - alpha * z_I;
}


//---------------------- DEHAZING FUNCTIONS -------------------//
// 朴素估计大气光图
void est_air(Mat& R, Mat& G, Mat& B, int s, double* A_r, double* A_g, double* A_b)
{
    double updateAlpha = 0.2;

    // 确保是 CV_64F
    if (R.type() != CV_64F) R.convertTo(R, CV_64F);
    if (G.type() != CV_64F) G.convertTo(G, CV_64F);
    if (B.type() != CV_64F) B.convertTo(B, CV_64F);

    // 暗通道
    Mat Im = min(min(R, G), B);

    // 模糊
    Mat blur_Im;
    blur(Im, blur_Im, Size(s, s), Point(-1, -1));

    // 找最大值位置
    int maxIdx[2] = {0, 0};
    minMaxIdx(blur_Im, NULL, NULL, NULL, maxIdx);

    int row = maxIdx[0];
    int col = maxIdx[1];

    // 访问像素值
    *A_r = R.at<double>(row, col) * updateAlpha + (1 - updateAlpha) * (*A_r);
    *A_g = G.at<double>(row, col) * updateAlpha + (1 - updateAlpha) * (*A_g);
    *A_b = B.at<double>(row, col) * updateAlpha + (1 - updateAlpha) * (*A_b);
}

void est_air_patchwise(const Mat& R, const Mat& G, const Mat& B,
                              int patchSize,
                              Mat& A_R, Mat& A_G, Mat& A_B) {
    // 初始化输出矩阵
    A_R = Mat::zeros(R.size(), CV_64F);
    A_G = Mat::zeros(G.size(), CV_64F);
    A_B = Mat::zeros(B.size(), CV_64F);

    for (int row = 0; row < R.rows; row += patchSize) {
        for (int col = 0; col < R.cols; col += patchSize) {
            int rowEnd = std::min(row + patchSize, R.rows);
            int colEnd = std::min(col + patchSize, R.cols);
            Rect patchROI(col, row, colEnd - col, rowEnd - row);

            // 取出 patch
            Mat patchR = R(patchROI);
            Mat patchG = G(patchROI);
            Mat patchB = B(patchROI);

            // 计算每个通道最大值
            double maxR, maxG, maxB;
            minMaxLoc(patchR, nullptr, &maxR);
            minMaxLoc(patchG, nullptr, &maxG);
            minMaxLoc(patchB, nullptr, &maxB);

            // 填充整个 patch
            A_R(patchROI).setTo(maxR);
            A_G(patchROI).setTo(maxG);
            A_B(patchROI).setTo(maxB);
        }
    }
    // --- 平滑处理，缓解块效应 ---
    int smoothKernel = patchSize * 2 + 1;  // 通常设为 patchSize 的两倍加一
    GaussianBlur(A_R, A_R, Size(smoothKernel, smoothKernel), 0);
    GaussianBlur(A_G, A_G, Size(smoothKernel, smoothKernel), 0);
    GaussianBlur(A_B, A_B, Size(smoothKernel, smoothKernel), 0);
}



// 估计透射率
Mat est_trans_fast(Mat& R, Mat& G, Mat& B, int s, double eeps, double k, Mat& A_r, Mat& A_g, Mat& A_b)
{

    /// Estimate transmission
    Mat R_n = R / A_r;
    Mat G_n = G / A_g;
    Mat B_n = B / A_b;


    Mat Im = min(min(R_n, G_n), B_n);
//    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "Mat rows: %d, cols: %d", Im.rows, Im.cols);

    eeps = (3 * 255 / mean(A_r + A_g + A_b).val[0]) * (3 * 255 / mean(A_r + A_g + A_b).val[0]) * eeps;
    double alpha = 2;
    int samplingRate = 4;
    Mat z_Im = staticMin(Im, s, eeps, alpha, samplingRate);

    return min(max(0.001, 1 - k * z_Im), 1);
}

// Remove haze with t & A
//dehazedImage = rmv_haze(R, G, B, t,air_r, air_g, air_b);
Mat rmv_haze(Mat& R, Mat& G, Mat& B, Mat& t, Mat& A_r, Mat& A_g, Mat& A_b)
{
    vector<Mat> channels(3);
    /// Remove haze
    channels[2] = (R - A_r) / t + A_r;
    channels[1] = (G - A_g) / t + A_g;
    channels[0] = (B - A_b) / t + A_b;

    Mat dst;
    merge(channels, dst);
    dst.convertTo(dst, CV_64F);
    return dst;
}
// 计算图像块的主成分方向
Vec3f computePatchLine(const Mat& patch) {
    if (patch.empty()) {
        throw std::runtime_error("Patch is empty");
    }

    // 如果 patch 不连续，先创建一个连续存储的副本
    Mat continuousPatch;
    if (!patch.isContinuous()) {
        continuousPatch = patch.clone();
    }
    else {
        continuousPatch = patch;
    }

    // 将 patch 展开为一行数据用于 PCA
    Mat reshaped = continuousPatch.reshape(1, continuousPatch.rows * continuousPatch.cols);
    // 检查展开后的矩阵
    if (reshaped.empty() || reshaped.cols != 3) {
        __android_log_print(ANDROID_LOG_INFO, "OpenCV", "reshaped.cols: %d", reshaped.cols);
        throw std::runtime_error("Reshaped patch is invalid or not 3-channel");
    }

    PCA pca(reshaped, Mat(), PCA::DATA_AS_ROW);
    Mat row = pca.eigenvectors.row(0); // 最大主成分方向
    Vec3f principalDirection = Vec3f(row.at<float>(0), row.at<float>(1), row.at<float>(2));
    return principalDirection;
}


// 选取符合条件的图像块
vector<Vec3f> selectPatches(const Mat& image, int patchSize) {
    //vector<Mat> patches;
    vector<Vec3f> lines;
    for (int y = 0; y <= image.rows - patchSize; y += patchSize) {
        for (int x = 0; x <= image.cols - patchSize; x += patchSize) {
            Mat patch = image(Rect(x, y, patchSize, patchSize));
            Vec3f line = computePatchLine(patch);
            if (norm(line) > 1e-2) { // 忽略强度不足的patch
                //patches.push_back(patch);
                lines.push_back(line);
            }
        }
    }
    return lines;
}

// 估计大气光方向
Vec3f estimateAtmosphericLightDirection(vector<Vec3f>& lines) {
    vector<Vec3f> directions;
    for (size_t i = 0; i < lines.size(); ++i) {
        //Vec3f direction = computePatchLine(patches[i]);
        Vec3f direction = lines[i];
        directions.push_back(direction / norm(direction));
    }
    Vec3f averageDirection(0, 0, 0);
    for (const auto& dir : directions) {
        averageDirection += dir;
    }
    return averageDirection / static_cast<float>(directions.size());
}

// 估计大气光大小
float estimateAtmosphericLightMagnitude(const Mat& image, const Vec3f& direction) {
    vector<double> brightnessValues;
    //double maxBrightness = 0.0;
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            Vec3f pixel = image.at<Vec3f>(y, x);
            double brightness = pixel.dot(direction);
            brightnessValues.push_back(brightness);
            /*maxBrightness = max(maxBrightness, brightness);*/
        }
    }

    std::sort(brightnessValues.begin(), brightnessValues.end());

    // 取第0.1百分位
    int percentileIndex = static_cast<int>(0.999 * brightnessValues.size());
    double topPercentileBrightness = brightnessValues[percentileIndex];

    return static_cast<float>(topPercentileBrightness);
}

// Estimate Atmospheric Light
void airlight_automatic_estimate(Mat& hazyImage) {
    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "Atmospheric Light est start");
    if (hazyImage.empty()) {
        return;
    }
    Mat transHazyImage;
    // 转换为浮点数类型以便计算
    hazyImage.convertTo(transHazyImage, CV_32FC3, 1.0 / 255);

    // 提取候选图像块
    int patchSize = 16;
    //vector<Mat> patches = selectPatches(transHazyImage, patchSize);
    vector<Vec3f> lines = selectPatches(transHazyImage, patchSize);

    // 估计大气光方向
    Vec3f atmosphericLightDirection = estimateAtmosphericLightDirection(lines);
    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "Atmospheric Light Direction over");
//    cout << "Atmospheric Light Direction: " << atmosphericLightDirection << endl;

    // 估计大气光大小
    float atmosphericLightMagnitude = estimateAtmosphericLightMagnitude(transHazyImage, atmosphericLightDirection);
    __android_log_print(ANDROID_LOG_INFO, "OpenCV", "Atmospheric Light Magnitude: %f", atmosphericLightMagnitude);

//    cout << "Atmospheric Light Magnitude: " << atmosphericLightMagnitude << endl;

    float rateOfNewAtm;
    if (frameCnt > 1) {
        rateOfNewAtm = 0.2;
    }
    else {
        rateOfNewAtm = 1;
    }
    air_r = min((min(atmosphericLightDirection[0] * atmosphericLightMagnitude * 255.0, 255.0)) * rateOfNewAtm + air_r * (1 - rateOfNewAtm), 255.0);
    air_g = min((min(atmosphericLightDirection[1] * atmosphericLightMagnitude * 255.0, 255.0)) * rateOfNewAtm + air_g * (1 - rateOfNewAtm), 255.0);
    air_b = min((min(atmosphericLightDirection[2] * atmosphericLightMagnitude * 255.0, 255.0)) * rateOfNewAtm + air_b * (1 - rateOfNewAtm), 255.0);
//    cout << "Atm. : ( " << air_r << ", " << air_g << ", " << air_b << " )" << endl;
}

Mat extract_blocks(const Mat& input_img, const Mat& guide_img, int patchSize) {
    priority_queue<tuple<int, int, double>, vector<tuple<int, int, double>>, BlockCompare> darkerQueue;   // 用于装小于阈值的块的值
    int threshold = 180;    // 阈值筛选
    int countOfBrightBlock = 0;
    float minPercentageOfBrightBlock = 0.1;

    int numSegRow = height / patchSize;
    int numSegCol = width / patchSize;
    int totalCountOfBlock = numSegCol * numSegRow;

    int minBrightBlock = (int)(minPercentageOfBrightBlock * totalCountOfBlock);

    // 初始化目标矩阵，用于存储满足条件的块
    Mat result;

    for (int i = 0; i < numSegCol; i++) {
        if (i * patchSize >= width) break;

        for (int j = 0; j < numSegRow; j++) {
            if (j * patchSize >= height) break;

            uchar guideValue = guide_img.at<uchar>(j, i);
            // 如果指导图的值大于thrshold，提取该块
            if (guideValue > threshold) {
                countOfBrightBlock++;
                Rect roi(patchSize * i, patchSize * j, patchSize, patchSize);
                Mat block = input_img(roi);

                // 将块水平拼接到结果矩阵中
                if (result.empty()) {
                    result = block.clone();
                }
                else {
                    hconcat(result, block, result);
                }
            } else {
                darkerQueue.emplace(i, j, guideValue);
            }
        }
    }

    while (countOfBrightBlock++ < minBrightBlock) {
        int i = get<0>(darkerQueue.top());
        int j = get<1>(darkerQueue.top());

        darkerQueue.pop();

        Rect roi(patchSize * i, patchSize * j, patchSize, patchSize);
        Mat block = input_img(roi);
        if (result.empty()) {
            result = block.clone();
        }
        else {
            hconcat(result, block, result);
        }
    }

    return result;
}