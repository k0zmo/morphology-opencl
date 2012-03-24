#include "cvutils.h"

namespace CvUtil {

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

void negateImage(cv::Mat& src)
{
	cv::Mat lut(1, 256, CV_8U);
	uchar* p = lut.ptr<uchar>();
	for(int i = 0; i < lut.cols; ++i)
	{
		*p++ = 255 - i;
	}
	cv::LUT(src, lut, src);
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

double scaleCoeff(const cv::Size& maxDstSize, const cv::Size& curSize)
{
	double fx = 1.0;

	if(curSize.width > maxDstSize.width || 
	   curSize.height > maxDstSize.height)
	{
		if(curSize.height > curSize.width)
			fx = static_cast<double>(maxDstSize.height) / curSize.height;
		else
			fx = static_cast<double>(maxDstSize.width) / curSize.width;
	}

	return fx;
}

void resizeWithAspect(cv::Mat& image, const cv::Size& dstSize)
{
	int xs = dstSize.width;
	int ys = dstSize.height;

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