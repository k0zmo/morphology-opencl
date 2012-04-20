#include "morphop.h"

namespace cvu {

const int OBJ = 255;
const int BCK = 0;

#ifdef _MSC_VER
	#define force_inline __forceinline
#elif defined(__GNUC__)
	#define force_inline inline __attribute__((always_inline))
#endif

int skeletonZHLutTable[256]  = {
	0,0,0,1,0,0,1,3,0,0,3,1,1,0,1,3,0,0,0,0,0,0,0,0,2,0,2,0,3,0,3,3,
	0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,3,0,2,2,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	2,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,3,0,0,0,0,0,0,0,3,0,0,0,3,0,2,0,
	0,0,3,1,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
	3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	2,3,1,3,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	2,3,0,1,0,0,0,1,0,0,0,0,0,0,0,0,3,3,0,1,0,0,0,0,2,2,0,0,2,0,0,0
};

static const float PI = 3.1415926535897932384626433832795f;
static const float RAD_PER_DEG = PI / 180.0f;
#define DEG2RAD(x) (x * RAD_PER_DEG)

cv::Mat standardStructuringElement(int xradius, int yradius,
	EStructuringElementType type, int rotation)
{
	cv::Point anchor(xradius, yradius);

	cv::Size elem_size(
		2 * anchor.x + 1,
		2 * anchor.y + 1);

	// OpenCV ma cos do rysowania elips ale nie spelnia oczekiwan
	// Patrz: przypadek elipsy o rozmiarze 17x31 i obrocie o 327 (wklesla)
	if(0 && type == SET_Ellipse)
	{
		int axis = std::max(xradius, yradius);
		cv::Size ksize(2*axis + 1, 2*axis + 1);
		cv::Mat element(ksize, CV_8U, cv::Scalar(0));

		cv::ellipse(element, cv::Point(axis, axis),
					cv::Size(xradius, yradius), -rotation,
					0, 360, cv::Scalar(255), -1, 8);
		return element;
	}

	if(type == SET_Ellipse)
	{
		// http://www.maa.org/joma/Volume8/Kalman/General.html
		//
		// (x*cos(t)+y*sin(t))^2   (x*sin(t)-y*cos(t))^2
		// --------------------- + --------------------- = 1
		//          a^2                     b^2

		double beta = -DEG2RAD(rotation);
		double sinbeta = sin(beta);
		double cosbeta = cos(beta);

		double a2 = static_cast<double>(xradius*xradius);
		double b2 = static_cast<double>(yradius*yradius);

		int axis = std::max(xradius, yradius);
		cv::Size ksize(2*axis + 1, 2*axis + 1);
		cv::Mat element(ksize, CV_8U);

		for(int y = -axis; y <= axis; ++y)
		{
			for(int x = -axis; x <= axis; ++x)
			{
				double n1 = x*cosbeta+y*sinbeta;
				double n2 = x*sinbeta-y*cosbeta;
				double lhs = (n1*n1)/a2 + (n2*n2)/b2;
				double rhs = 1;
				element.at<uchar>(y+axis, x+axis) = (lhs <= rhs ? 1 : 0);
			}
		}
		return element;
	}

	int shape;
	switch(type)
	{
	case SET_Rect: shape = cv::MORPH_RECT; break;
	//case SET_Ellipse: shape = cv::MORPH_ELLIPSE; break;
	case SET_Cross: shape = cv::MORPH_CROSS; break;
	default: return cv::Mat();
	}

	cv::Mat element = cv::getStructuringElement(shape, elem_size, anchor);
	return rotateStructuringElement(rotation, element);
}

cv::Mat rotateStructuringElement(int rotation, const cv::Mat& _element)
{
	// Rotacja elementu strukturalnego
	if(rotation == 0)
		return _element;

	rotation %= 360;

	auto rotateImage = [](const cv::Mat& source, double angle) -> cv::Mat
	{
		cv::Point2f srcCenter(source.cols/2.0f, source.rows/2.0f);
		cv::Mat rotMat = cv::getRotationMatrix2D(srcCenter, angle, 1.0f);
		cv::Mat dst;
		cv::warpAffine(source, dst, rotMat, source.size(), cv::INTER_LINEAR);
		return dst;
	};

	int s = 2 * std::max(_element.rows, _element.cols);
	int b = s/4;

	cv::Mat tmp(cv::Size(s, s), CV_8U, cv::Scalar(0));
	cv::copyMakeBorder(_element, tmp, b,b,b,b, cv::BORDER_CONSTANT);
	cv::Mat element(rotateImage(tmp, rotation));

	// Trzeba teraz wyciac niepotrzebny nadmiar pikseli ramkowych
	int top = 0, 
		bottom = element.rows, 
		left = 0,
		right = element.cols;

	// Zwraca true jesli wskazany wiersz w danej macierzy nie jest "pusty" (nie ma samych 0)
	auto checkRow = [](int row, const cv::Mat& e) -> bool
	{
		const uchar* p = e.ptr<uchar>(row);
		for(int x = 0; x < e.cols; ++x)
			if(p[x] != 0)
				return true;
		return false;
	};

	// Zwraca true jesli wskazana kolumna w danej macierzy nie jest "pusta" (nie ma samych 0)
	auto checkColumn = [](int column, const cv::Mat& e) -> bool
	{
		for(int y = 0; y < e.rows; ++y)
			if(e.at<uchar>(y, column) != 0)
				return true;
		return false;
	};

	// Kadruj gore
	for(int y = 0; y < element.rows; ++y)
	{
		if (checkRow(y, element))
			break;
		++top;
	}

	// Kadruj dol
	for(int y = element.rows-1; y >= 0; --y)
	{
		if (checkRow(y, element))
			break;
		--bottom;
	}

	// Kadruj lewa strone
	for(int x = 0; x < element.cols; ++x)
	{
		if (checkColumn(x, element))
			break;
		++left;
	}

	// Kadruj prawa strone
	for(int x = element.cols-1; x >= 0; --x)
	{
		if (checkColumn(x, element))
			break;
		--right;
	}

	int width = right-left;
	int height = bottom-top;

	// Zalozenie jest ze element strukturalny ma rozmiar 2n+1,
	// ponizsze dwa bloki strzega tego warunku

	if(!(width % 2))
	{
		width++; 
		// jesli wyjdziemy za zakres to zmniejsz poczatek ROI
		if(left+width > element.cols) 
			--left;
	}
	if(!(height % 2))
	{
		height++;
		// jesli wyjdziemy za zakres to zmniejsz poczatek ROI
		if(top+height > element.rows) 
			--top;
	}

	return element(cv::Rect(left, top, width, height));
}

void hitmissOutline(const cv::Mat& src, cv::Mat& dst)
{
	dst = src.clone();

	// 1 - obiekt (bialy)
	// 0 - tlo (czarny)
	// X - dowolny

	// Element strukturalny
	// 1|1|1
	// 1|X|1
	// 1|1|1
	//

	const uchar* pixels2 = src.ptr<uchar>();
	uchar* pixels = dst.ptr<uchar>();
	int rowOffset = src.cols;

	#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; ++y)
	{
		int offset = 1 + y * rowOffset;
		for(int x = 1; x < src.cols - 1; ++x)
		{
			//if (src.at<uchar>(y-1, x-1) == OBJ &&
			//	src.at<uchar>(y-1, x  ) == OBJ &&
			//	src.at<uchar>(y-1, x+1) == OBJ &&
			//	src.at<uchar>(y  , x-1) == OBJ &&
			//	src.at<uchar>(y  , x+1) == OBJ &&
			//	src.at<uchar>(y+1, x-1) == OBJ &&
			//	src.at<uchar>(y+1, x  ) == OBJ &&
			//	src.at<uchar>(y+1, x+1) == OBJ)
			//{
			//	dst.at<uchar>(y, x) = BCK;
			//}

			uchar p1 = pixels2[offset-rowOffset-1];
			uchar p2 = pixels2[offset-rowOffset];
			uchar p3 = pixels2[offset-rowOffset+1];
			uchar p4 = pixels2[offset-1];
			uchar p5 = pixels2[offset];
			uchar p6 = pixels2[offset+1];
			uchar p7 = pixels2[offset+rowOffset-1];
			uchar p8 = pixels2[offset+rowOffset];
			uchar p9 = pixels2[offset+rowOffset+1];

			uchar v = p5;

			if(v == OBJ)
			{
				if (p1==OBJ && p2==OBJ && p3==OBJ && p4==OBJ && 
					p6==OBJ && p7==OBJ && p8==OBJ && p9==OBJ)
				{
					v = BCK;
				}
			}			

			pixels[offset++] = v;
		}
	}
}

force_inline int _skeleton_iter1(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//

	int d = 0;

	#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == OBJ &&
				src.at<uchar>(y-1, x  ) == OBJ &&
				src.at<uchar>(y-1, x+1) == OBJ &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y+1, x-1) == BCK &&
				src.at<uchar>(y+1, x  ) == BCK &&
				src.at<uchar>(y+1, x+1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
				d++;
			}
		}
	}
	return d;
}

force_inline int _skeleton_iter2(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny pierwszy - 90 w lewo
	// 1|X|0
	// 1|1|0
	// 1|x|0
	//

	int d = 0;

	#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == OBJ &&
				src.at<uchar>(y-1, x+1) == BCK &&
				src.at<uchar>(y  , x-1) == OBJ &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == BCK &&
				src.at<uchar>(y+1, x-1) == OBJ &&
				src.at<uchar>(y+1, x+1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
				d++;
			}
		}
	}
	return d;
}

force_inline int _skeleton_iter3(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny pierwszy - 180 w lewo
	// 0|0|0
	// X|1|X
	// 1|1|1
	//

	int d = 0;

	#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == BCK &&
				src.at<uchar>(y-1, x  ) == BCK &&
				src.at<uchar>(y-1, x+1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y+1, x-1) == OBJ &&
				src.at<uchar>(y+1, x  ) == OBJ &&
				src.at<uchar>(y+1, x+1) == OBJ)
			{
				dst.at<uchar>(y, x) = BCK;
				d++;
			}
		}
	}
	return d;
}

force_inline int _skeleton_iter4(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny pierwszy - 270 w lewo
	// 0|X|1
	// 0|1|1
	// 0|X|1
	//

	int d = 0;

	#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == BCK &&
				src.at<uchar>(y-1, x+1) == OBJ &&
				src.at<uchar>(y  , x-1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == OBJ &&
				src.at<uchar>(y+1, x-1) == BCK &&
				src.at<uchar>(y+1, x+1) == OBJ)
			{
				dst.at<uchar>(y, x) = BCK;
				d++;
			}
		}
	}
	return d;
}

force_inline int _skeleton_iter5(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny drugi
	// X|1|X
	// 0|1|1
	// 0|0|X
	//

	int d = 0;

	#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x  ) == OBJ &&
				src.at<uchar>(y  , x-1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == OBJ &&
				src.at<uchar>(y+1, x-1) == BCK &&
				src.at<uchar>(y+1, x  ) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
				d++;
			}
		}
	}

	return d;
}

force_inline int _skeleton_iter6(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny drugi - 90 stopni w lewo
	// X|1|X
	// 1|1|0
	// X|0|0
	//

	int d = 0;

	#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x  ) == OBJ &&
				src.at<uchar>(y  , x-1) == OBJ &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == BCK &&
				src.at<uchar>(y+1, x  ) == BCK &&
				src.at<uchar>(y+1, x+1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
				d++;
			}
		}
	}
	return d;
}

force_inline int _skeleton_iter7(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny drugi - 180 stopni w lewo
	// X|0|0
	// 1|1|0
	// X|1|X
	//

	int d = 0;

	#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x  ) == BCK &&
				src.at<uchar>(y-1, x+1) == BCK &&
				src.at<uchar>(y  , x-1) == OBJ &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == BCK &&
				src.at<uchar>(y+1, x  ) == OBJ)
			{
				dst.at<uchar>(y, x) = BCK;
				d++;
			}
		}
	}

	return d;
}

force_inline int _skeleton_iter8(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|1
	// X|1|X
	//

	int d = 0;

	#pragma omp parallel for
	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == BCK &&
				src.at<uchar>(y-1, x  ) == BCK &&
				src.at<uchar>(y  , x-1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == OBJ &&
				src.at<uchar>(y+1, x  ) == OBJ)
			{
				dst.at<uchar>(y, x) = BCK;
				d++;
			}
		}
	}
	return d;
}

int hitmissSkeleton(const cv::Mat& _src, cv::Mat &dst)
{
	int niters = 0;

	cv::Mat src = _src.clone();
	dst = src.clone();

	while(true) 
	{
		// iteracja
		++niters;
		int d = 0;

		d += _skeleton_iter1(src, dst);
		dst.copyTo(src);

		d += _skeleton_iter2(src, dst);
		dst.copyTo(src);

		d += _skeleton_iter3(src, dst);
		dst.copyTo(src);

		d += _skeleton_iter4(src, dst);
		dst.copyTo(src);

		d += _skeleton_iter5(src, dst);
		dst.copyTo(src);

		d += _skeleton_iter6(src, dst);
		dst.copyTo(src);

		d += _skeleton_iter7(src, dst);
		dst.copyTo(src);

		d += _skeleton_iter8(src, dst);

		printf("Iteration: %3d, pixel changed: %5d\r", niters, d);

		if(d == 0)
			break;

		dst.copyTo(src);
	}
	printf("\n");

	return niters;
}

int hitmissSkeletonZhangSuen(const cv::Mat& src, cv::Mat& dst)
{
	// Based on ImageJ implementation of skeletonization which is
	// based on an a thinning algorithm by by Zhang and Suen (CACM, March 1984, 236-239)
	int pass = 0;
	int pixelsRemoved = 0;
	dst = src.clone();
	int niters = 0;

	auto thin = [&dst](int pass) -> int
	{
		const int bgColor = 0;

		cv::Mat tmp = dst.clone();
		uchar* pixels2 = tmp.ptr<uchar>();
		uchar* pixels = dst.ptr<uchar>();
		int pixelsRemoved = 0;
		int xMin = 1, yMin = 1, rowOffset = dst.cols;

		#pragma omp parallel for
		for(int y = yMin; y < dst.rows - 1; ++y)
		{
			int offset = xMin + y * rowOffset;
			for(int x = xMin; x < dst.cols - 1; ++x)
			{
				uchar p5 = pixels2[offset];
				uchar v = p5;
				if(v != bgColor)
				{
					uchar p1 = pixels2[offset-rowOffset-1];
					uchar p2 = pixels2[offset-rowOffset];
					uchar p3 = pixels2[offset-rowOffset+1];
					uchar p4 = pixels2[offset-1];
					uchar p6 = pixels2[offset+1];
					uchar p7 = pixels2[offset+rowOffset-1];
					uchar p8 = pixels2[offset+rowOffset];
					uchar p9 = pixels2[offset+rowOffset+1];

					// lut index
					int index = 
						((p4&0x01) << 7) |
						((p7&0x01) << 6) |
						((p8&0x01) << 5) |
						((p9&0x01) << 4) |
						((p6&0x01) << 3) |
						((p3&0x01) << 2) |
						((p2&0x01) << 1) |
						(p1&0x01);
					int code = skeletonZHLutTable[index];

					//odd pass
					if((pass & 1) == 1)
					{ 
						if(code == 2 || code == 3)
						{
							v = bgColor;
							pixelsRemoved++;
						}
					} 
					//even pass
					else
					{
						if (code == 1 || code == 3)
						{
							v = bgColor;
							pixelsRemoved++;
						}
					}
				}
				pixels[offset++] = v;
			}
		}

		return pixelsRemoved;
	};

	do 
	{
		niters++;
		pixelsRemoved  = thin(pass++);
		pixelsRemoved += thin(pass++);
		printf("Iteration: %3d, pixel changed: %5d\r", niters, pixelsRemoved);
	} while (pixelsRemoved > 0);
	printf("\n");

	return niters;
}

int morphEx(const cv::Mat& src, cv::Mat& dst,
	EMorphOperation op, const cv::Mat& se)
{
	int iters = 1;

	// Operacje hit-miss
	if (op == MO_Outline ||
		op == MO_Skeleton ||
		op == MO_Skeleton_ZhangSuen)
	{
		switch (op)
		{
		case MO_Outline:
			{
				hitmissOutline(src, dst);
				break;
			}
		case MO_Skeleton:
			{
				iters = hitmissSkeleton(src, dst);
				break;
			}
		case MO_Skeleton_ZhangSuen:
			{
				iters = hitmissSkeletonZhangSuen(src, dst);
				break;
			}
		default: break;
		}
	}
	else
	{
		int op_type;
		switch(op)
		{
		case MO_Erode: op_type = cv::MORPH_ERODE; break;
		case MO_Dilate: op_type = cv::MORPH_DILATE; break;
		case MO_Open: op_type = cv::MORPH_OPEN; break;
		case MO_Close: op_type = cv::MORPH_CLOSE; break;
		case MO_Gradient: op_type = cv::MORPH_GRADIENT; break;
		case MO_TopHat: op_type = cv::MORPH_TOPHAT; break;
		case MO_BlackHat: op_type = cv::MORPH_BLACKHAT; break;
		default: op_type = cv::MORPH_ERODE; break;
		}

		if(se.rows == 0 || se.cols == 0)
			return 0;

		cv::morphologyEx(src, dst, op_type, se);
	}

	return iters;
}

} // end of namespace
