#include "morphop.h"

const int OBJ = 255;
const int BCK = 0;

#ifdef _MSC_VER
#define force_inline __forceinline
#elif defined(__GNUC__)
#define force_inline inline __attribute__((always_inline))
#endif

int lutTable[256]  = {
	0,0,0,1,0,0,1,3,0,0,3,1,1,0,1,3,0,0,0,0,0,0,0,0,2,0,2,0,3,0,3,3,
	0,0,0,0,0,0,0,0,3,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,3,0,2,2,
	0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	2,0,0,0,0,0,0,0,2,0,0,0,2,0,0,0,3,0,0,0,0,0,0,0,3,0,0,0,3,0,2,0,
	0,0,3,1,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,
	3,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	2,3,1,3,0,0,1,3,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,
	2,3,0,1,0,0,0,1,0,0,0,0,0,0,0,0,3,3,0,1,0,0,0,0,2,2,0,0,2,0,0,0
};

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
cv::Mat standardStructuringElement(int xradius, int yradius,
	EStructureElementType type, int rotation)
{
	cv::Point anchor(xradius, yradius);

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
		element = structuringElementDiamond(std::min(anchor.x, anchor.y));
	}

	// Rotacja elementu strukturalnego
	if(rotation != 0)
	{
		rotation %= 360;

		auto rotateImage = [](const cv::Mat& source, double angle) -> cv::Mat
		{
			cv::Point2f srcCenter(source.cols/2.0f, source.rows/2.0f);
			cv::Mat rotMat = cv::getRotationMatrix2D(srcCenter, angle, 1.0f);
			cv::Mat dst;
			cv::warpAffine(source, dst, rotMat, source.size(), cv::INTER_NEAREST);
			return dst;
		};

		int s = 2 * std::max(element.rows, element.cols);
		int b = s/4;

		cv::Mat tmp(cv::Size(s, s), CV_8U, cv::Scalar(0));
		cv::copyMakeBorder(element, tmp, b,b,b,b, cv::BORDER_CONSTANT);
		element = rotateImage(tmp, rotation);

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

		element = element(cv::Rect(left, top, right-left, bottom-top));
	}

	return element;
}
// -------------------------------------------------------------------------
void morphologyOutline(const cv::Mat& src, cv::Mat& dst)
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
// -------------------------------------------------------------------------
force_inline int _morphologySkeleton_iter1(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny pierwszy
	// 1|1|1
	// X|1|X
	// 0|0|0
	//

	int d = 0;

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
// -------------------------------------------------------------------------
force_inline int _morphologySkeleton_iter2(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny pierwszy - 90 w lewo
	// 1|X|0
	// 1|1|0
	// 1|x|0
	//

	int d = 0;

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
// -------------------------------------------------------------------------
force_inline int _morphologySkeleton_iter3(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny pierwszy - 180 w lewo
	// 0|0|0
	// X|1|X
	// 1|1|1
	//

	int d = 0;

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
// -------------------------------------------------------------------------
force_inline int _morphologySkeleton_iter4(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny pierwszy - 270 w lewo
	// 0|X|1
	// 0|1|1
	// 0|X|1
	//

	int d = 0;

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
// -------------------------------------------------------------------------
force_inline int _morphologySkeleton_iter5(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny drugi
	// X|1|X
	// 0|1|1
	// 0|0|X
	//

	int d = 0;

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
// -------------------------------------------------------------------------
force_inline int _morphologySkeleton_iter6(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny drugi - 90 stopni w lewo
	// X|1|X
	// 1|1|0
	// X|0|0
	//

	int d = 0;

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
// -------------------------------------------------------------------------
force_inline int _morphologySkeleton_iter7(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny drugi - 180 stopni w lewo
	// X|0|0
	// 1|1|0
	// X|1|X
	//

	int d = 0;

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
// -------------------------------------------------------------------------
force_inline int _morphologySkeleton_iter8(const cv::Mat& src, cv::Mat& dst)
{
	// Element strukturalny drugi - 270 stopni w lewo
	// 0|0|X
	// 0|1|1
	// X|1|X
	//

	int d = 0;

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
// -------------------------------------------------------------------------
int morphologySkeleton(const cv::Mat& _src, cv::Mat &dst)
{
	int niters = 0;

	cv::Mat src = _src.clone();
	dst = src.clone();

	while(true) 
	{
		// iteracja
		++niters;
 		int d = 0;

		d += _morphologySkeleton_iter1(src, dst);
		dst.copyTo(src);

		d += _morphologySkeleton_iter2(src, dst);
		dst.copyTo(src);

		d += _morphologySkeleton_iter3(src, dst);
		dst.copyTo(src);

		d += _morphologySkeleton_iter4(src, dst);
		dst.copyTo(src);

		d += _morphologySkeleton_iter5(src, dst);
		dst.copyTo(src);

		d += _morphologySkeleton_iter6(src, dst);
		dst.copyTo(src);

		d += _morphologySkeleton_iter7(src, dst);
		dst.copyTo(src);

		d += _morphologySkeleton_iter8(src, dst);
		
		printf("Iteration: %3d, pixel changed: %5d\r", niters, d);

		if(d == 0)
			break;

		dst.copyTo(src);
	}

	return niters;
}

int morphologySkeletonZhangSuen(const cv::Mat& _src, cv::Mat& dst)
{
	// Based on ImageJ implementation of skeletonization which is
	// based on an a thinning algorithm by by Zhang and Suen (CACM, March 1984, 236-239)
	int pass = 0;
	int pixelsRemoved = 0;
	dst = _src.clone();
	int niters = 0;

	auto thin = [&dst](int pass) -> int
	{
		const int bgColor = 0;

		cv::Mat tmp = dst.clone();
		uchar* pixels2 = tmp.ptr<uchar>();
		uchar* pixels = dst.ptr<uchar>();
		int pixelsRemoved = 0;
		int xMin = 1, yMin = 1, rowOffset = dst.cols;

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
					int code = lutTable[index];

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

	return niters;
}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// Odpowiedniki funkcji opencl'a

void morphologyErode(const cv::Mat& src, cv::Mat& dst, const cv::Mat& element)
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

#if 1
	// Obraz docelowy
	dst = cv::Mat(src.size(), CV_8U, cv::Scalar(erodeINF));

	// kwintesencja funkcji - filtrowanie
	#pragma omp parallel for
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
	
#if 0
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
	dst = cv::Mat(src.size(), CV_8U, cv::Scalar(0));
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
	struct cl_int2 { int s[2]; };
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
