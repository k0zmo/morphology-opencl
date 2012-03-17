#pragma once

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace CvUtil {

void negateImage(cv::Mat& src);

}