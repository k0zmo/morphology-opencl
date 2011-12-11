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
	OT_Outline,
	OT_Skeleton,
	OT_Skeleton_ZhangSuen
};

enum EStructuringElementType
{
	SET_Rect,
	SET_Ellipse,
	SET_Cross,
	SET_Diamond
};

// Zwraca element strukturalny
cv::Mat standardStructuringElement(int xradius, int yradius, 
	EStructuringElementType type, int rotation = 0);

// Operacja morfologiczna - erozja
void morphologyErode(const cv::Mat& src, cv::Mat& dst, const cv::Mat& element);
// Operacja morfologiczna - kontrur
void morphologyOutline(const cv::Mat& src, cv::Mat& dst);
// Operacja morfologiczna - szkieletyzacja
int morphologySkeleton(const cv::Mat &src, cv::Mat &dst);
// Operacja morfologiczna - Zhang and Suen
int morphologySkeletonZhangSuen(const cv::Mat& _src, cv::Mat& dst);

// Tablica LUT do szkieletyzacji Zhang'a i Suen'a
extern int lutTable[256];