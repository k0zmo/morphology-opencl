#pragma once

#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>

namespace Morphology {

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
	OT_Skeleton_ZhangSuen,
	OT_None
};

enum EStructuringElementType
{
	SET_Rect,
	SET_Ellipse,
	SET_Cross,
	SET_Custom
};

enum EBayerCode
{
	BC_None,
	BC_RedGreen,
	BC_GreenRed,
	BC_BlueGreen,
	BC_GreenBlue
};

// Zwraca element strukturalny
cv::Mat standardStructuringElement(int xradius, int yradius, 
	EStructuringElementType type, int rotation = 0);

// Obraca dany element strukturalny
cv::Mat rotateStructuringElement(int rotation, const cv::Mat& _element);

int process(const cv::Mat& src, cv::Mat& dst,
	EOperationType op, const cv::Mat& se = cv::Mat());

// Operacja morfologiczna - kontrur
void outline(const cv::Mat& src, cv::Mat& dst);

// Operacja morfologiczna - szkieletyzacja
int skeleton(const cv::Mat& src, cv::Mat& dst);

// Operacja morfologiczna - Zhang and Suen
int skeletonZhangSuen(const cv::Mat& src, cv::Mat& dst);

extern int skeletonZHLutTable[256];

}