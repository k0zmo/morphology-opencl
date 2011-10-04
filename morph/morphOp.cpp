#define CV_NO_BACKWARD_COMPATIBILITY

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Obiekt - bialy
static const int OBJ = 255;
// Tlo - czarne
static const int BCK = 0;

// -------------------------------------------------------------------------
cv::Mat structuringElementDiamond(int radius)
{
	int a = radius;
	int s = 2 * radius + 1;

	cv::Mat element = cv::Mat(s, s, CV_8U, cv::Scalar(1));

	// top-left
	int y = a;
	for(int j = 0; j < a; ++j)
	{
		for(int i = 0; i < y; ++i)
		{
			element.at<uchar>(j, i) = 0;
		}
		--y;
	}


	// top-right
	y = a + 1;
	for(int j = 0; j < a; ++j)
	{
		for(int i = y; i < s; ++i)
		{
			element.at<uchar>(j, i) = 0;
		}
		++y;
	}

	// bottom-left
	y = 1;
	for(int j = a; j < s; ++j)
	{
		for(int i = 0; i < y; ++i)
		{
			element.at<uchar>(j, i) = 0;
		}
		++y;
	}

	// bottom-right
	y = s - 1;
	for(int j = a; j < s; ++j)
	{
		for(int i = y; i < s; ++i)
		{
			element.at<uchar>(j, i) = 0;
		}
		--y;
	}

	return element;
}
// -------------------------------------------------------------------------
int countDiffPixels(const cv::Mat& src1, const cv::Mat& src2)
{
	cv::Mat diff;
	cv::compare(src1, src2, diff, cv::CMP_NE);
	return cv::countNonZero(diff);
}
// -------------------------------------------------------------------------
void morphologyRemove(const cv::Mat& src, cv::Mat& dst)
{
	// TODO: border

	// 1 - obiekt (bialy)
	// 0 - tlo (czarny)
	// X - dowolny

	// Element strukturalny
	// 1|1|1
	// 1|X|1
	// 1|1|1
	//

	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == OBJ &&
				src.at<uchar>(y-1, x  ) == OBJ &&
				src.at<uchar>(y-1, x+1) == OBJ &&
				src.at<uchar>(y  , x-1) == OBJ &&
				src.at<uchar>(y  , x+1) == OBJ &&
				src.at<uchar>(y+1, x-1) == OBJ &&
				src.at<uchar>(y+1, x  ) == OBJ &&
				src.at<uchar>(y+1, x+1) == OBJ)
			{
				dst.at<uchar>(y, x) = BCK;
			}
		}
	}
}
// -------------------------------------------------------------------------
void _morphologySkeleton_iter(cv::Mat src, cv::Mat dst)
{
	// TODO: border

	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//

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
			}
		}
	}

	src = dst.clone();

	// Element strukturalny pierwszy - 90 w lewo
	// 1|X|0
	// 1|1|0
	// 1|x|0
	//

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
			}
		}
	}

	src = dst.clone();

	// Element strukturalny pierwszy - 180 w lewo
	// 0|0|0
	// X|1|X
	// 1|1|1
	//

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
			}
		}
	}

	src = dst.clone();

	// Element strukturalny pierwszy - 270 w lewo
	// 0|X|1
	// 0|1|1
	// 0|X|1
	//

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
			}
		}
	}

	src = dst.clone();

	// Element strukturalny drugi
	// X|1|X
	// 0|1|1
	// 0|0|X
	//

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
			}
		}
	}

	src = dst.clone();

	// Element strukturalny drugi - 90 stopni w lewo
	// X|1|X
	// 1|1|0
	// X|0|0
	//

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
			}
		}
	}

	src = dst.clone();

	// Element strukturalny drugi - 180 stopni w lewo
	// X|0|0
	// 1|1|0
	// X|1|X
	//

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
			}
		}
	}

	src = dst.clone();

	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|1
	// X|1|X
	//

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
			}
		}
	}
}
// -------------------------------------------------------------------------
void _morphologyPruning_iter(cv::Mat src, cv::Mat dst)
{
	// TODO: border

	// Element strukturalny pierwszy
	// 0|X|X
	// 0|1|0
	// 0|0|0
	//

	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == BCK &&
				src.at<uchar>(y  , x-1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == BCK &&
				src.at<uchar>(y+1, x-1) == BCK &&
				src.at<uchar>(y+1, x  ) == BCK &&
				src.at<uchar>(y+1, x+1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
			}
		}
	}
	src = dst.clone();

	// Element strukturalny pierwszy - 90 stopni w lewo
	// X|0|0
	// X|1|0
	// 0|0|0
	//

	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x  ) == BCK &&
				src.at<uchar>(y-1, x+1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == BCK &&
				src.at<uchar>(y+1, x-1) == BCK &&
				src.at<uchar>(y+1, x  ) == BCK &&
				src.at<uchar>(y+1, x+1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
			}
		}
	}

	src = dst.clone();

	// Element strukturalny pierwszy - 180 stopni w lewo
	// 0|0|0
	// 0|1|0
	// X|X|0
	//

	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == BCK &&
				src.at<uchar>(y-1, x  ) == BCK &&
				src.at<uchar>(y-1, x+1) == BCK &&
				src.at<uchar>(y  , x-1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == BCK &&
				src.at<uchar>(y+1, x+1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
			}
		}
	}

	src = dst.clone();

	// Element strukturalny pierwszy - 270 stopni w lewo
	// 0|0|0
	// 0|1|X
	// 0|0|X
	//

	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == BCK &&
				src.at<uchar>(y-1, x  ) == BCK &&
				src.at<uchar>(y-1, x+1) == BCK &&
				src.at<uchar>(y  , x-1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y+1, x-1) == BCK &&
				src.at<uchar>(y+1, x  ) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
			}
		}
	}

	src = dst.clone();

	// Element strukturalny drugi
	// X|X|0
	// 0|1|0
	// 0|0|0
	//

	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x+1) == BCK &&
				src.at<uchar>(y  , x-1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == BCK &&
				src.at<uchar>(y+1, x-1) == BCK &&
				src.at<uchar>(y+1, x  ) == BCK &&
				src.at<uchar>(y+1, x+1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
			}
		}
	}

	src = dst.clone();

	// Element strukturalny drugi - 90 stopni w lewo
	// 0|0|0
	// X|1|0
	// X|0|0
	//

	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == BCK &&
				src.at<uchar>(y-1, x  ) == BCK &&
				src.at<uchar>(y-1, x+1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == BCK &&
				src.at<uchar>(y+1, x  ) == BCK &&
				src.at<uchar>(y+1, x-1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
			}
		}
	}

	src = dst.clone();

	// Element strukturalny drugi - 180 stopni w lewo
	// 0|0|0
	// 0|1|0
	// 0|X|X
	//

	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == BCK &&
				src.at<uchar>(y-1, x  ) == BCK &&
				src.at<uchar>(y-1, x+1) == BCK &&
				src.at<uchar>(y  , x-1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y  , x+1) == BCK &&
				src.at<uchar>(y+1, x-1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
			}
		}
	}

	src = dst.clone();

	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|X
	// 0|0|0
	//

	for(int y = 1; y < src.rows - 1; ++y)
	{
		for(int x = 1; x < src.cols - 1; ++x)
		{
			if (src.at<uchar>(y-1, x-1) == BCK &&
				src.at<uchar>(y-1, x  ) == BCK &&
				src.at<uchar>(y  , x-1) == BCK &&
				src.at<uchar>(y  , x  ) == OBJ &&
				src.at<uchar>(y+1, x-1) == BCK &&
				src.at<uchar>(y+1, x  ) == BCK &&
				src.at<uchar>(y+1, x+1) == BCK)
			{
				dst.at<uchar>(y, x) = BCK;
			}
		}
	}
}
// -------------------------------------------------------------------------
int morphologySkeleton(cv::Mat &src, cv::Mat &dst) 
{
	int niters = 0;

	while(true) 
	{
		// iteracja
		_morphologySkeleton_iter(src, dst);
		++niters;

		// warunek stopu
		if(countDiffPixels(src, dst) == 0) break;

		src = dst.clone();
	}

	return niters;
}
// -------------------------------------------------------------------------
int morphologyVoronoi(cv::Mat &src, cv::Mat &dst, int prune) 
{
	int niters = 0;

	// Diagram voronoi jest operacja dualna do szkieletowania
	src = 255 - src;
	dst = 255 - dst;

	while(true) 
	{
		// iteracja
		_morphologySkeleton_iter(src, dst);
		++niters;

		// warunek stopu
		if(countDiffPixels(src, dst) == 0) break;

		src = dst.clone();
	}

	if(prune > 0)
	{
		for(int i = 0; i < prune; ++i)
		{
			src = dst.clone();

			// iteracja
			_morphologyPruning_iter(src, dst);
			++niters;				
		}
	}

	return niters;
}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Odpowiedniki funkcji opencl'a

void doErode(
	const cv::Mat& src,
	cv::Mat& dst,
	const cv::Mat& element)
{
	static const uchar erodeINF = 255;

	// wskaznik na poczatek macierzy wejsciowej
	const uchar* pInput = src.ptr<uchar>();
	// szerokosc macierzy wejsciowej
	int imx = src.cols;
	// wysokosc macierzy wejsciowej
	int imy = src.rows;

	const uchar* pSe = element.ptr<uchar>();
	// szerokosc elementu strukturalnego
	int sex = element.cols;
	// wysokosc elementu strukturalego
	int sey = element.rows;

	// srodek w poziomie elementu strukturalnego
	int anchorX = (element.cols - 1) / 2;
	// srodek w pionie elementu strukturalnego
	int anchorY = (element.rows - 1) / 2;

	// Rozmiar obramowania w poziomie i pionie
	int borderX = anchorX;
	int borderY = anchorY;

	// rozmiar poszerzonego obrazu
	size_t tempx = borderX * 2 + imx;
	size_t tempy = borderY * 2 + imy;

	//temp8 - obraz poszerzony o obramowanie o wartosci erodeINF
	uchar* tempSrc = new uchar[tempx * tempy];
	memset(tempSrc, erodeINF, tempx * tempy * sizeof(uchar));

	dst = cv::Mat(src.size(), CV_8U, cv::Scalar(erodeINF));
	uchar* pOutput = dst.ptr<uchar>();
	//memset(pOutput, erodeINF, imx * imy * sizeof(uchar));

	// Skopiuj dane pikseli z obrazu zrodlowego do tymczasowego z dod. obramowaniem
	const uchar* scanLineIn = pInput;
	uchar* scanLineOut = tempSrc + tempx*borderY;	

	for (int y = 0; y < imy; ++y)
	{
		memcpy(scanLineOut + borderX, scanLineIn, imx * sizeof(uchar));

		scanLineIn += imx; // kolejny scanline z obrazu zrodlowego
		scanLineOut += tempx; // kolejny scanline z obrazu docelowego (kapke 'dluzszy')
	}

	// kwintesencja funkcji - filtrowanie
	for(int m = 0; m < sey; ++m)
	{
		for(int n = 0; n < sex; ++n)
		{
			if(pSe[m * element.cols + n] != 0)
			{
				int distx = n - anchorX;
				int disty = m - anchorY;
				int d = borderX + distx;

				//uchar* scanLineOut = pOutput;

				for(int yy = borderY, y = 0; yy < borderY + imy; ++yy, ++y)
				{
					const uchar* scanLineIn = tempSrc + tempx*(yy+disty) + d /*borderX + distx*/;
					uchar* scanLineOut = pOutput + imx*y;

					for(int xx = 0; xx < imx; ++xx)
					{
						if(*scanLineOut > *scanLineIn)
							*scanLineOut = *scanLineIn;

						*scanLineOut++;
						*scanLineIn++;
					}

					//scanLineOut += imx;
				}
			}
		}
	}

	// Obraz tymczasowy zwalniamu
	delete[] tempSrc;


#if 0
	static const uchar erodeINF = 255;

	// wskaznik na poczatek macierzy wejsciowej
	const uchar* pInput = src.ptr<uchar>();
	// szerokosc macierzy wejsciowej
	int imx = src.cols;
	// wysokosc macierzy wejsciowej
	int imy = src.rows;

	const uchar* pSe = element.ptr<uchar>();
	// szerokosc elementu strukturalnego
	int sex = element.cols;
	// wysokosc elementu strukturalego
	int sey = element.rows;

	// srodek w poziomie elementu strukturalnego
	int anchorX = (element.cols - 1) / 2;
	// srodek w pionie elementu strukturalnego
	int anchorY = (element.rows - 1) / 2;

	int borderX = anchorX;
	int borderY = anchorY;

	uchar* pOutput = dst.ptr<uchar>();

	size_t tempx = borderX * 2 + imx;
	size_t tempy = borderY * 2 + imy;

	uchar* temp8 = new uchar[tempx * tempy];
	uchar* temp_out8 = new uchar[tempx * tempy];

	memset(temp8, erodeINF, tempx * tempy * sizeof(uchar));
	memset(temp_out8, erodeINF, tempx * tempy * sizeof(uchar));

	{
		uchar* out_scan8 = temp8 + tempx*borderY;
		const uchar* in_scan8 = pInput;

		for (int y = 0; y < imy; ++y)
		{
			memcpy(out_scan8 + borderX, in_scan8, imx * sizeof(uchar));

			in_scan8 += imx;
			out_scan8 += tempx;
		}
	}


	for(int m = 0; m < sey; ++m)
	{
		for(int n = 0; n < sex; ++n)
		{
			if(pSe[m * element.cols + n] != 0)
			{
				int distx = n - anchorX;
				int disty = m - anchorY;

				for(int yy = borderY; yy < borderY + imy; ++yy)
				{
					const uchar* in_scan8 = temp8 + tempx*(yy+disty) + borderX + distx;
					uchar* out_scan8 = temp_out8 + tempx*yy + borderX;

					for(int xx = 0; xx < imx; ++xx)
					{
						if(*out_scan8 > *in_scan8)
							*out_scan8 = *in_scan8;

						*out_scan8++;
						*in_scan8++;
					}
				}
			}
		}
	}

	uchar* out8 = pOutput;
	{
		uchar* in_scan8 = temp_out8 + tempx * borderY;
		uchar* out_scan8 = out8;

		for (int y = 0; y < imy; ++y)
		{   
			memcpy(out_scan8, in_scan8 + borderX, imx * sizeof(uchar));

			in_scan8 += tempx;
			out_scan8 += imx;
		}
	}

	delete[] temp8;
	delete[] temp_out8;

#endif
#if 0
	const uchar erodeINF = 255;

	int anchorX = (element.cols - 1) / 2;
	int anchorY = (element.rows - 1) / 2;

	const uchar* pInput = src.ptr<uchar>();
	const uchar* pFilter = element.ptr<uchar>();
	uchar* pOutput = dst.ptr<uchar>();

	for(int yy = anchorY; yy < src.rows - anchorY; ++yy)
	{
		for(int xx = anchorX; xx < src.cols - anchorX; ++xx)
		{
			uchar v = erodeINF;
			const uchar* pF = pFilter;

			for(int r = -1 * anchorY; r < (anchorY + 1); ++r)
			{
				const size_t idIn = (yy + r) * src.cols + xx;

				for(int c = -1 * anchorX; c < (anchorX + 1); ++c)
				{
					const size_t ii = idIn + c;

					if(*pF++ != 0)
						v = min(v, pInput[ii]);
				}
			}

			pOutput[yy * dst.cols + xx] = v;
		}
	}
#endif
}

void doDilate(
	const cv::Mat& src,
	cv::Mat& dst,
	const cv::Mat& element)
{
	const uchar dilateINF = 255;

	int anchorX = (element.cols - 1) / 2;
	int anchorY = (element.rows - 1)/ 2;

	int gwidth = src.cols;
	int gheight = src.rows;

	for(int yy = 0; yy < src.rows; ++yy)
	{
		for(int xx = 0; xx < src.cols; ++xx)
		{
			uchar val = dilateINF;
			size_t eid = 0;
			const uchar* pelement = element.ptr<uchar>();

			for(int y = -1 * anchorY; y < (anchorY + 1); ++y)
			{
				for(int x = -1 * anchorX; x < (anchorX + 1); ++x)
				{
					int xi = xx + x;
					int yi = yy + y;

					if(xi < 0 || xi >= gwidth || yi < 0 || yi >= gheight)
					{
						if(pelement[eid] != 0)
							val = std::max(val, dilateINF);
					}

					else if(pelement[eid] != 0)
					{
						val = std::max(val, src.at<uchar>(yi, xi));
					}

					++eid;
				}
			}

			dst.at<uchar>(yy, xx) = val;
		}
	}
}