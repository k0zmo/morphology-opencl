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
	const uchar erodeINF = 255;

	int anchorX = element.cols / 2;
	int anchorY = element.rows / 2;

	int gwidth = src.cols;
	int gheight = src.rows;

	for(int yy = 0; yy < src.rows; ++yy)
	{
		for(int xx = 0; xx < src.cols; ++xx)
		{
			uchar val = erodeINF;
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
							val = min(val, erodeINF);
					}

					else if(pelement[eid] != 0)
					{
						val = min(val, src.at<uchar>(yi, xi));
					}

					++eid;
				}
			}

			dst.at<uchar>(yy, xx) = val;
		}
	}
}

void doDilate(
	const cv::Mat& src,
	cv::Mat& dst,
	const cv::Mat& element)
{
	const uchar dilateINF = 255;

	int anchorX = element.cols / 2;
	int anchorY = element.rows / 2;

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
							val = max(val, dilateINF);
					}

					else if(pelement[eid] != 0)
					{
						val = max(val, src.at<uchar>(yi, xi));
					}

					++eid;
				}
			}

			dst.at<uchar>(yy, xx) = val;
		}
	}
}