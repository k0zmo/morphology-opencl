#include "morphOp.h"

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
cv::Mat standardStructuringElement(int xsize, int ysize,
	EStructureElementType type, int rotation)
{
	cv::Point anchor(xsize, ysize);

	cv::Size elem_size(
		2 * anchor.x + 1,
		2 * anchor.y + 1);

	cv::Mat element;

	if(type == SET_Rect)
	{
		element = cv::getStructuringElement(cv::MORPH_RECT, elem_size, anchor);
	}
	else if(type == SET_Ellipse)
	{
		element = cv::getStructuringElement(cv::MORPH_ELLIPSE, elem_size, anchor);
	}
	else if(type == SET_Cross)
	{
		element = cv::getStructuringElement(cv::MORPH_CROSS, elem_size, anchor);
	}
	else
	{
		element = structuringElementDiamond(qMin(anchor.x, anchor.y));
	}

	// Rotacja elementu strukturalnego
	if(rotation != 180)
	{
		int angle = rotation;
		if(angle >= 180) { angle -= 180; }
		else { angle += 180; }
		angle = 360 - angle;

		auto rotateImage = [=](const cv::Mat& source, double angle) -> cv::Mat
		{
			cv::Point2f srcCenter(source.cols/2.0f, source.rows/2.0f);
			cv::Mat rotMat = cv::getRotationMatrix2D(srcCenter, angle, 1.0f);
			cv::Mat dst;
			cv::warpAffine(source, dst, rotMat, source.size(), cv::INTER_NEAREST);
			return dst;
		};

		//cv::Mat tmp(element.size() * 2, CV_8U, cv::Scalar(0));
		//int border = element.rows/4;
		//cv::copyMakeBorder(element, tmp, border, border, border, border, cv::BORDER_CONSTANT);

		element = rotateImage(element, angle);

		//for(int i = 0; i < element.cols*element.rows; ++i)
		//{ uchar* p = element.ptr<uchar>(); if(p[i] == 1) p[i] = 255; }
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

void doErode(const cv::Mat& src, cv::Mat& dst, const cv::Mat& element)
{
	static const uchar erodeINF = 255;

	// srodek w poziomie elementu strukturalnego
	int anchorX = (element.cols - 1) / 2;
	// srodek w pionie elementu strukturalnego
	int anchorY = (element.rows - 1) / 2;

	// Rozmiar obramowania w poziomie i pionie
	int borderX = anchorX;
	int borderY = anchorY;

	// rozmiar poszerzonego obrazu
	size_t tempx = borderX * 2 + src.cols;
	size_t tempy = borderY * 2 + src.rows;

	//temp8 - obraz poszerzony o obramowanie o wartosci erodeINF
	uchar* tempSrc = new uchar[tempx * tempy];
	memset(tempSrc, erodeINF, tempx * tempy * sizeof(uchar));

	// Skopiuj dane pikseli z obrazu zrodlowego do tymczasowego z dod. obramowaniem
	const uchar* scanLineIn = src.ptr<uchar>();
	uchar* scanLineOut = tempSrc + tempx*borderY;	

	for (int y = 0; y < src.rows; ++y)
	{
		memcpy(scanLineOut + borderX, scanLineIn, src.cols * sizeof(uchar));

		scanLineIn += src.cols; // kolejny scanline z obrazu zrodlowego
		scanLineOut += tempx; // kolejny scanline z obrazu docelowego (kapke 'dluzszy')
	}

#if 0
	// Obraz docelowy
	dst = cv::Mat(src.size(), CV_8U, cv::Scalar(erodeINF));

	// kwintesencja funkcji - filtrowanie
	for(int m = 0; m < element.rows; ++m)
	{
		for(int n = 0; n < element.cols; ++n)
		{
			const uchar* se = element.ptr<uchar>();

			if(se[m * element.cols + n] != 0)
			{
				int distx = n - anchorX;
				int disty = m - anchorY;
				int d = borderX + distx;

				uchar* pDst = dst.ptr<uchar>();

				for(int yy = borderY; yy < borderY + src.rows; ++yy)
				{
					const uchar* pSrc = tempSrc + tempx*(yy+disty) + d;

					int xx = 0;
					for( ; xx <= src.cols - 4; xx += 4)
					{
						pDst[0] = std::min(pDst[0], pSrc[0]);
						pDst[1] = std::min(pDst[1], pSrc[1]);
						pDst[2] = std::min(pDst[2], pSrc[2]);
						pDst[3] = std::min(pDst[3], pSrc[3]);

						pDst += 4;
						pSrc += 4;
					}

					for( ; xx < src.cols; ++xx)
					{
						pDst[0] = std::min(pDst[0], pSrc[0]);

						pSrc++;
						pDst++;
					}
				}
			}
		}
	}

	// Obraz tymczasowy zwalniamu
	delete[] tempSrc;
#endif
	
#if 1
	// Wydobadz wspolrzedne 'aktywne' z elementu strukturalnego
	std::vector<cv::Point> coords;
	for(int y = 0; y < element.rows; ++y)
	{
		const uchar* krow = element.data + element.step*y;

		for(int x = 0; x < element.cols; ++x)
		{
			if(krow[x] == 0)
				continue;

			coords.emplace_back(cv::Point(x, y));
		}
	}
	std::vector<const uchar*> ptrs(coords.size());

	// Wskazniki na kazdy z rzedow obrazu
	std::vector<const uchar*> rows(tempy);
	for(size_t i = 0; i < rows.size(); ++i)
		rows[i] = tempSrc + tempx*i;

	// Obraz docelowy
	dst = cv::Mat(src.size(), CV_8U, cv::Scalar(erodeINF));
	uchar* pDst = dst.ptr<uchar>();
	size_t dstep = dst.cols;

	// Filtracja
	for(int y = 0; y < src.rows; ++y, pDst += dstep)
	{
		const uchar** srcs = &rows[y];
		const uchar** kp = &ptrs[0];
		int nz = static_cast<int>(coords.size());

		for(int k = 0; k < nz; ++k)
			kp[k] = srcs[coords[k].y] + coords[k].x;

		int x = 0;
		for( ; x <= src.cols - 4; x += 4 )
		{
			const uchar* sptr = kp[0] + x;
			uchar s0 = sptr[0], s1 = sptr[1], s2 = sptr[2], s3 = sptr[3];

			for(int k = 1; k < nz; k++ )
			{
				sptr = kp[k] + x;
				s0 = std::min(s0, sptr[0]); s1 = std::min(s1, sptr[1]);
				s2 = std::min(s2, sptr[2]); s3 = std::min(s3, sptr[3]);
			}

			pDst[x] = s0; pDst[x+1] = s1;
			pDst[x+2] = s2; pDst[x+3] = s3;
		}

		for(; x < src.cols; ++x)
		{
			uchar s0 = kp[0][x];
			for(int k = 1; k < nz; ++k)
				s0 = std::min(s0, kp[k][x]);
			pDst[x] = s0;
		}
	}

#endif

#if 0
	std::vector<cl_int2> coords;
	for(int y = 0; y < element.rows; ++y)
	{
		const uchar* krow = element.data + element.step*y;

		for(int x = 0; x < element.cols; ++x)
		{
			if(krow[x] == 0)
				continue;

			cl_int2 c = {x - anchorX, y - anchorY};
			coords.push_back(c);
		}
	}

	const uchar* input = src.ptr<uchar>();
	dst = cv::Mat(src.size(), CV_8U, cv::Scalar(0));
	size_t rowPitch = src.cols;

	for(int y = anchorY; y < src.rows - anchorY; ++y)
	{
		for(int x = anchorX; x < src.cols - anchorX; ++x)
		{
			uchar val = erodeINF;

			for(int i = 0; i < coords.size(); ++i)
			{
				int yy = coords[i].s[1] + y;
				int xx = coords[i].s[0] + x;
				val = std::min(val, input[xx + yy * rowPitch]);
			}

			uchar* pdst = dst.ptr<uchar>();
			pdst[x + y * rowPitch] = val;
		}
	}
#endif
}
