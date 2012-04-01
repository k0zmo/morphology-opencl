#pragma once

#include "oclfilter.h"

class oclBayerFilter : public oclFilter
{
public:
	oclBayerFilter(oclContext* ctx);

	virtual double run();
	void setBayerFilter(cvu::EBayerCode bc);

private:
	cl::Kernel kernels[4];
	cl::Kernel* kernel;
};