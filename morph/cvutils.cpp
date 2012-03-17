#include "cvutils.h"

namespace CvUtil {

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

} // end of namespace