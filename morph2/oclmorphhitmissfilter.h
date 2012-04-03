#pragma once

#include "oclfilter.h"
#include "morphop.h"

class oclMorphHitMissFilter : public oclFilter
{
public:
	oclMorphHitMissFilter(oclContext* ctx, bool atomicCounters);

	void setHitMissOperation(cvu::EMorphOperation op);
	cvu::EMorphOperation hitMissOperation() const
	{ return hmOp; }

	virtual double run();

private:
	cl::Kernel kernelOutline;
	cl::Kernel kernelSkeleton_pass[2];
	cl::Kernel kernelSkeleton_iter[8];

	cvu::EMorphOperation hmOp;

private:
	double runHitMissKernel(cl::Kernel* kernel,
		const oclImage2DHolder& source, oclImage2DHolder& output,
		const oclBufferHolder* lut = nullptr,
		oclBufferHolder* atomicCounter = nullptr);
};

