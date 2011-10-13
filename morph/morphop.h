#pragma once

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/imgproc/imgproc.hpp>

// Obiekt - bialy
extern const int OBJ;
// Tlo - czarne
extern const int BCK;

enum EOperationType
{
	OT_Erode,
	OT_Dilate,
	OT_Open,
	OT_Close,
	OT_Gradient,
	OT_TopHat,
	OT_BlackHat,
	OT_Thinning,
	OT_Skeleton,
	OT_Voronoi
};

enum EStructureElementType
{
	SET_Rect,
	SET_Ellipse,
	SET_Cross,
	SET_Diamond
};

// Zwraca element strukturalny
cv::Mat standardStructuringElement(int xradius, int yradius, 
	EStructureElementType type, int rotation = 0);

// Operacja morfologiczna - erozja
void morphologyErode(const cv::Mat& src, cv::Mat& dst, const cv::Mat& element);

// Operacja morfologiczna - scienienie

void morphologyThinning(const cv::Mat& src, cv::Mat& dst);

// Operacja morfologiczna - szkieletyzacja
int morphologySkeleton(cv::Mat &src, cv::Mat &dst);

// Operacja morfologiczna - diagram Voronoi
int morphologyVoronoi(cv::Mat &src, cv::Mat &dst, int prune);
