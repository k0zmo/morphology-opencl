#include "cvutils.h"

#include <QImage>

namespace cvu {

QImage toQImage(const cv::Mat& image)
{
	switch(image.type())
	{
	case CV_8UC1:
		return QImage(
			reinterpret_cast<const quint8*>(image.data),
			image.cols, image.rows, image.step, 
			QImage::Format_Indexed8);
	case CV_8UC4:
		{
			QImage img(reinterpret_cast<const quint8*>(image.data),
				image.cols, image.rows, image.step, 
				QImage::Format_RGB888);
			return img.rgbSwapped();
		}
	}

	qDebug("Can't convert image of type %d", image.type());
	return QImage();
}

void negate(const cv::Mat& src, cv::Mat& dst)
{
	//cv::LUT(_src, lut, _dst);
	cv::Mat lut(1, 256, CV_8U);
	uchar* p = lut.ptr<uchar>();

	for(int i = 0; i < lut.cols; ++i)
		*p++ = 255 - i;

	cv::LUT(src, lut, dst);
}

void bayerFilter(const cv::Mat& src, cv::Mat& dst, EBayerCode bc)
{
	// Jest bug dla CV_BayerXX2GRAY i trzeba wykonac sciezke okrezna
	switch(bc)
	{
	case BC_RedGreen:  cv::cvtColor(src, dst, CV_BayerRG2BGR); break;
	case BC_GreenRed:  cv::cvtColor(src, dst, CV_BayerGR2BGR); break;
	case BC_BlueGreen: cv::cvtColor(src, dst, CV_BayerBG2BGR); break;
	case BC_GreenBlue: cv::cvtColor(src, dst, CV_BayerGB2BGR); break;
	default: break;
	}
	cvtColor(dst, dst, CV_BGR2GRAY);
}

void convert01To0255(cv::Mat& src)
{
	cv::Mat lut(1, 256, CV_8U);
	uchar* p = lut.ptr<uchar>();
	p[0] = 0;
	for(int i = 1; i < lut.cols; ++i) 
		p[i] = 255;
	cv::LUT(src, lut, src);
}

std::pair<double, double> scaleCoeffs(const cv::Size& curSize, const cv::Size& dstSize)
{
	// Pierwszy przypadek: 512x512 -> 256x256
	int xs = qMin(dstSize.width, curSize.width);
	int ys = qMin(dstSize.height, curSize.height);

	if(xs == curSize.width && ys == curSize.height)
		return std::make_pair(1.0, 1.0);

	// Drugi przypadek: 512x256 -> 256x256 (256x128)
	if(curSize.width != curSize.height)
	{
		if(curSize.width > curSize.height)
			ys = dstSize.width  / (static_cast<double>(curSize.width) /
								   static_cast<double>(curSize.height));
		else
			xs = dstSize.height / (static_cast<double>(curSize.height) /
								   static_cast<double>(curSize.width));
	}
	// Trzeci przypadek: 512x512 -> 256x128 (128x128)
	else if(dstSize.width != dstSize.height)
	{
		ys = xs = qMin(xs, ys);
	}

	double fx = static_cast<double>(xs) / static_cast<double>(curSize.width);
	double fy = static_cast<double>(ys) / static_cast<double>(curSize.height);

	return std::make_pair(fx, fy);
}

void fitImageToSize(cv::Mat& image, const cv::Size& dstSize)
{
	auto coeffs = scaleCoeffs(image.size(), dstSize);
	double fx = coeffs.first;
	double fy = coeffs.second;

	cv::resize(image, image, cv::Size(), fx, fy, cv::INTER_LINEAR);
}

void fitImageToWholeSpace(cv::Mat& image, const cv::Size& sizeHint)
{
	int xs = sizeHint.width;
	int ys = sizeHint.height;

	if(image.cols != image.rows)
	{
		auto round = [](double v) { return static_cast<int>(v + 0.5); };

		if(image.cols > image.rows)
			ys = image.rows * round((double)xs/image.cols);
		else
			xs = image.cols * round((double)ys/image.rows);
	}

	cv::resize(image, image, cv::Size(xs, ys), 0.0, 0.0, cv::INTER_NEAREST);
}

} // end of namespace
