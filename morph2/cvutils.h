#pragma once

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <QImage>

namespace cvu {

enum EBayerCode
{
	BC_None,
	BC_RedGreen,
	BC_GreenRed,
	BC_BlueGreen,
	BC_GreenBlue
};

// Konwersja cv::Mat do QImage
// Obslugiwane formaty: CV_8UC1 oraz CV_8UC3
QImage toQImage(const cv::Mat& image);

// Negacja danego obrazu (w miejscu)
void negate(const cv::Mat& src, cv::Mat& dst);
void bayerFilter(const cv::Mat& src, cv::Mat& dst, EBayerCode bc);

// Konwertuje 0, 1 na 0,255 (w miejscu)
void convert01To0255(cv::Mat& src);

double scaleCoeff(const cv::Size& maxDstSize, const cv::Size& curSize);
void resizeWithAspect(cv::Mat& image, const cv::Size& dstSize);

}