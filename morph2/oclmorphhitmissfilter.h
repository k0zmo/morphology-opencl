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

	qreal copyImage2D(const QCLImage2D& src, QCLImage2D& dst);
	qreal readAtomicCounter(QCLBuffer& buf, cl_uint& dst);
	qreal zeroAtomicCounter(QCLBuffer& buf);
};

