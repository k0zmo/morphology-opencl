#include "stdafx.h"
#define CV_NO_BACKWARD_COMPATIBILITY

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

// Obiekt - bialy
static const int OBJ = 255;
// Tlo - czarne
static const int BCK = 0;

// -------------------------------------------------------------------------
void morphRemove(const cv::Mat& src, cv::Mat& dst)
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
void morphSkeleton(cv::Mat src, cv::Mat dst)
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
void morphPruning(cv::Mat src, cv::Mat dst)
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