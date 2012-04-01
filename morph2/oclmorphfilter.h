#pragma once

#include "oclfilter.h"
#include "morphop.h"

class oclMorphFilter : public oclFilter
{
public:
	oclMorphFilter(oclContext* ctx);

	void setMorphologyOperation(cvu::EMorphOperation op);
	cvu::EMorphOperation morphologyOperation() const
	{ return morphOp; }

	virtual double run();

	void setStructuringElement(const cv::Mat& selement);

private:
	cl::Kernel kernelErode;
	cl::Kernel kernelDilate;
	cl::Kernel kernelGradient;
	cl::Kernel kernelSubtract;

	oclBufferHolder structuringElement;
	int seRadiusX;
	int seRadiusY;

	cvu::EMorphOperation morphOp;

private:

	double runMorphologyKernel(cl::Kernel* kernel,
		const oclImage2DHolder& source,
		oclImage2DHolder& output);
	
	double runSubtractKernel(
		const oclImage2DHolder& sourceA,
		const oclImage2DHolder& sourceB,
		oclImage2DHolder& output);
};