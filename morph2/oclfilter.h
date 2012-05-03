#pragma once

#include "oclcontext.h"
#include "cvutils.h"

class oclFilter
{
public:
	oclFilter(oclContext* ctx);
	virtual ~oclFilter();
	
	virtual double run() = 0;

	void prepareDestinationHolder();
	void finishUpDestinationHolder();

	virtual void setSourceImage(const oclImage2DHolder& src,
		const cv::Rect& roi = cvu::WholeImage);

	void setOutputDeviceImage(const oclImage2DHolder& img);
	void unsetOutputDeviceImage() { ownsOutput = true; }

	oclImage2DHolder outputDeviceImage() const { return dst; }

protected:
	cl::NDRange computeOffset(int minBorderX, int minBorderY);
	cl::NDRange computeGlobal(int minBorderX, int minBorderY);

protected:
	oclContext* ctx;

	const oclImage2DHolder* src;
	oclImage2DHolder dst;
	bool ownsOutput;
	cv::Rect roi;
};
