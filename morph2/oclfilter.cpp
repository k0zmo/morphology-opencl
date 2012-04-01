#include "oclfilter.h"

oclFilter::oclFilter(oclContext* ctx)
	: ctx(ctx)
	, src(nullptr)
	, roi(cvu::WholeImage)
{
}

oclFilter::~oclFilter()
{
}

void oclFilter::setSourceImage(
	const oclImage2DHolder& src,
	const cv::Rect& roi)
{
	this->src = &src;

	if(roi == cvu::WholeImage)
	{
		this->roi.width = src.width;
		this->roi.height = src.height;
	}
	else
	{
		// TODO
		//	roi.x = std::min(roi.x, src.width);
		//	roi.y = std::min(roi.y, src.height);
		//	roi.width = [something]
	}
}

cl::NDRange oclFilter::computeOffset(
	int minBorderX, int minBorderY)
{
	// TODO jesli roi jest mniejszy niz obraz nie trzeba go przesuwac

	cv::Rect r(roi);
	r.x = std::max(minBorderX, r.x);
	r.y = std::max(minBorderY, r.y);

	return cl::NDRange(r.x, r.y);
}

cl::NDRange oclFilter::computeGlobal(
	int minBorderX, int minBorderY)
{
	// TODO jesli roi jest inny niz WholeImage

	cl::NDRange local = ctx->workgroupSize();

	int gx = oclContext::roundUp(roi.width - 2*minBorderX, local[0]);
	int gy = oclContext::roundUp(roi.height - 2*minBorderY, local[1]);

	return cl::NDRange(gx, gy);
}

