#pragma once

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace cvu {

enum EMorphOperation
{
	MO_Erode,
	MO_Dilate,
	MO_Open,
	MO_Close,
	MO_Gradient,
	MO_TopHat,
	MO_BlackHat,
	MO_Outline,
	MO_Skeleton,
	MO_Skeleton_ZhangSuen,
	MO_None
};

inline bool isHitMiss(EMorphOperation mo)
{
	return mo == MO_Outline ||
		mo == MO_Skeleton ||
		mo == MO_Skeleton_ZhangSuen;
}

enum EStructuringElementType
{
	SET_Rect,
	SET_Ellipse,
	SET_Cross,
	SET_Custom
};

// Zwraca element strukturalny
cv::Mat standardStructuringElement(int xradius, int yradius, 
	EStructuringElementType type, int rotation = 0);

// Obraca dany element strukturalny
cv::Mat rotateStructuringElement(int rotation, const cv::Mat& element);

int morphEx(const cv::Mat& src, cv::Mat& dst,
	EMorphOperation op, const cv::Mat& se = cv::Mat());

// Operacja morfologiczna - kontrur
void hitmissOutline(const cv::Mat& src, cv::Mat& dst);

// Operacja morfologiczna - szkieletyzacja
int hitmissSkeleton(const cv::Mat& src, cv::Mat& dst);

// Operacja morfologiczna - Zhang and Suen
int hitmissSkeletonZhangSuen(const cv::Mat& src, cv::Mat& dst);

extern int skeletonZHLutTable[256];

}