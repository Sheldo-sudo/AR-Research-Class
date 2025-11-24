#include "threshold.h"

Mat threshold_image_output(const Mat &input_img, int patchSize) {

    int height = input_img.rows;
    int width = input_img.cols;//图像的长宽

    int img_type[10] = {};//图片种类
    int cnt_1[10] = {};//图像等级对应像素总数

    Mat gray_img;
    cvtColor(input_img, gray_img, COLOR_RGB2GRAY);//创建原图的灰度图
    Mat gray_ave_img;
    gray_ave_img.create(width / patchSize, height / patchSize, CV_8UC1);//对比的平均灰度新空白图,为四个像素的压缩图片

    int num_segrow = height / patchSize;
    int num_segcol = width / patchSize;//将图像分为4*4小块,

    /*if (patchSize * num_segrow != height) {
        num_segrow++;
    }

    if (patchSize * num_segcol != width) {
        num_segcol++;
    }*/

    //printf("%d %d\n", height, width);

    for (int i = 0; i < num_segcol; i++) {
        if (i * patchSize >= width) break;
        int rect_height = 0;
        int rect_width = 0;//小方框长宽设定

        rect_width = (width - i * patchSize < patchSize) ? (width - i * patchSize)
                                                         : patchSize;//剩余方框检测

        int num = 1;

        for (int j = 0; j < num_segrow; j++) {

            if (j * patchSize >= height) break;
            rect_height = (height - j * patchSize < patchSize) ? (height - j * patchSize)
                                                               : patchSize;
            Rect roi(patchSize * i, patchSize * j, rect_width, rect_height);//划定小方框4*4

            Mat roi_img = gray_img(roi);

            double total_gray = 0.0;
            int pixel_count = rect_height * rect_width;

            for (int y = 0; y < rect_width; y++) {
                for (int x = 0; x < rect_height; x++) {
                    total_gray += roi_img.at<uchar>(y, x);//计算小块中总灰度
                }
            }

            double ave_gray = total_gray / pixel_count;//计算小块平均灰度

            for (num = 1; num <= 10; num++) {
                if (ave_gray >= 0 * num && ave_gray <= 25.5 * num) {
                    cnt_1[num - 1]++;
                    gray_ave_img.at<uchar>(i, j) = 25.5 * num;
                    break;//灰度等级个数信息，灰度图生成信息  //注意i, j 是对应原图的坐标/4的值
                } else {
                    num++;
                }
            }
        }
        num = 1;
    }
    return gray_ave_img;
//	imwrite("gray_ave_img.jpg", output_threshold_img);
}
