#pragma once

#include "oclfilter.h"
#include "morphop.h"

class oclMorphHitMissFilter : public oclFilter
{
	Q_DISABLE_COPY(oclMorphHitMissFilter)
public:
	oclMorphHitMissFilter(QCLContext* ctx, bool atomicCounters);

	void setHitMissOperation(cvu::EMorphOperation op);
	cvu::EMorphOperation hitMissOperation() const
	{ return hmOp; }

	virtual qreal run();
	
private:
	QCLKernel kernelOutline;
	QCLKernel kernelSkeleton_pass[2];
	QCLKernel kernelSkeleton_iter[8];

	cvu::EMorphOperation hmOp;

	QCLBuffer atomicCounter;
	QCLBuffer zhLut;

private:
	qreal runHitMissKernel(QCLKernel* kernel,
		const QCLImage2D& source, QCLImage2D& output,
		const QCLBuffer* lut = nullptr,
		QCLBuffer* atomicCounter = nullptr);
};

class oclMorphHitMissFilterBuffer : public oclFilterBuffer
{
	Q_DISABLE_COPY(oclMorphHitMissFilterBuffer)
public:
	oclMorphHitMissFilterBuffer(QCLContext* ctx, 
		bool localMemory, bool atomicCounters);

	void setHitMissOperation(cvu::EMorphOperation op);
	cvu::EMorphOperation hitMissOperation() const
	{ return hmOp; }

	virtual qreal run();

private:
	QCLKernel kernelOutline;
	QCLKernel kernelSkeleton_pass[2];
	QCLKernel kernelSkeleton_iter[8];

	cvu::EMorphOperation hmOp;

	QCLBuffer atomicCounter;
	QCLBuffer zhLut;

private:
	qreal runHitMissKernel(QCLKernel* kernel,
		const QCLBuffer& source, QCLBuffer& output,
		const QCLBuffer* lut = nullptr,
		QCLBuffer* atomicCounter = nullptr);
};

