#pragma once

#include "oclfilter.h"

class oclBayerFilter : public oclFilter
{
	Q_DISABLE_COPY(oclBayerFilter)
public:
	oclBayerFilter(QCLContext* ctx);

	virtual qreal run();
	void setBayerFilter(cvu::EBayerCode bc);
private:
	QCLKernel d_kernels[4];
	QCLKernel* d_kernel;
};

class oclBayerFilterBuffer : public oclFilterBuffer
{
	Q_DISABLE_COPY(oclBayerFilterBuffer)
public:
	oclBayerFilterBuffer(QCLContext* ctx);

	virtual qreal run();
	void setBayerFilter(cvu::EBayerCode bc);
private:
	QCLKernel d_kernels[4];
	QCLKernel* d_kernel;
};

