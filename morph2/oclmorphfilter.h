#pragma once

#include "oclfilter.h"
#include "morphop.h"

class oclMorphFilter : public oclFilter
{
	Q_DISABLE_COPY(oclMorphFilter)
public:
	oclMorphFilter(QCLContext* ctx,
		const char *erode, const char *dilate,
		const char *gradient);

	void setStructuringElement(const cv::Mat& selement);
	void setMorphologyOperation(cvu::EMorphOperation op);
	cvu::EMorphOperation morphologyOperation() const
	{ return morphOp; }

	virtual qreal run();

private:
	QCLKernel kernelErode;
	QCLKernel kernelDilate;
	QCLKernel kernelGradient;
	QCLKernel kernelSubtract;

	QCLBuffer structuringElement;
	int seRadiusX;
	int seRadiusY;
	int csize;

	cvu::EMorphOperation morphOp;

private:
	qreal runMorphologyKernel(QCLKernel* kernel,
		const QCLImage2D& source,
		QCLImage2D& output);
	
	qreal runSubtractKernel(
		const QCLImage2D& sourceA,
		const QCLImage2D& sourceB,
		QCLImage2D& output);
};

class oclMorphFilterBuffer : public oclFilterBuffer
{
	Q_DISABLE_COPY(oclMorphFilterBuffer)
public:
	oclMorphFilterBuffer(QCLContext* ctx,
		const char *erode, const char *dilate,
		const char *gradient, const char* subtract);

	void setStructuringElement(const cv::Mat& selement);
	void setMorphologyOperation(cvu::EMorphOperation op);
	cvu::EMorphOperation morphologyOperation() const
	{ return morphOp; }

	virtual qreal run();

private:
	QCLKernel kernelErode;
	QCLKernel kernelDilate;
	QCLKernel kernelGradient;
	QCLKernel kernelSubtract;

	QCLBuffer structuringElement;
	int seRadiusX;
	int seRadiusY;
	int csize;

	cvu::EMorphOperation morphOp;

private:
	qreal runMorphologyKernel(QCLKernel* kernel,
		const QCLBuffer& source,
		QCLBuffer& output);

	qreal runSubtractKernel(
		const QCLBuffer& sourceA,
		const QCLBuffer& sourceB,
		QCLBuffer& output);
};