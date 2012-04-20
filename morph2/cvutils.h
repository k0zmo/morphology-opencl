#pragma once

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

class QImage;

namespace cvu {

// Filtr Bayer'a
enum EBayerCode
{
	BC_None,
	BC_RedGreen,
	BC_GreenRed,
	BC_BlueGreen,
	BC_GreenBlue
};

void bayerFilter(const cv::Mat& src, cv::Mat& dst, EBayerCode bc);

// Konwersja cv::Mat do QImage
// Obslugiwane formaty: CV_8UC1 oraz CV_8UC3
QImage toQImage(const cv::Mat& image);

// Negacja danego obrazu (w miejscu)
void negate(const cv::Mat& src, cv::Mat& dst);

// Konwertuje obraz w postaci 0,1 na 0,255 (w miejscu)
void convert01To0255(cv::Mat& src);

std::pair<double, double> scaleCoeffs(const cv::Size& curSize, const cv::Size& dstSize);
void fitImageToSize(cv::Mat& image, const cv::Size& dstSize);
void fitImageToWholeSpace(cv::Mat& image, const cv::Size& sizeHint);

static const cv::Rect WholeImage(0, 0, -1, -1);
}
