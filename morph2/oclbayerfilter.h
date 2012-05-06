#pragma once

//#include "oclfilter.h"

#include "cvutils.h"

#include <QCLContext>
#include <QCLEvent>

QCLImageFormat morphImageFormat();

//class oclFilter
//{
//public:
//	oclFilter(QCLContext* ctx);
//	virtual ~oclFilter();

//	virtual QCLEvent run() = 0;

//	virtual void setSourceImage(const QCLImage2D& src);
//		//const QRect& roi = cvu::WholeImage);

//	//void setOutputDeviceImage(const oclImage2DHolder& img);
//	//void unsetOutputDeviceImage() { ownsOutput = true; }

//	QCLImage2D outputDeviceImage() const { return d_dst; }

//	QCLWorkSize localWorkSize() const { return d_localSize; }
//	void setLocalWorkSize(const QCLWorkSize& local) { d_localSize = local; }

//protected:
//	QCLWorkSize computeOffset(int minBorderX, int minBorderY);
//	QCLWorkSize computeGlobal(int minBorderX, int minBorderY);

//	void prepareDestinationHolder();
//	void finishUpDestinationHolder();

//protected:
//	QCLContext* d_ctx;
//	QCLWorkSize d_localSize;

//	const QCLImage2D* d_src;
//	QCLImage2D d_dst;

//	//QRect roi;
//	//bool ownsOutput;
//};


//class oclBayerFilter : public oclFilter
//{
//public:
//	oclBayerFilter(QCLContext* ctx);

//	virtual QCLEvent run();
//	void setBayerFilter(cvu::EBayerCode bc);

//private:
//	QCLKernel d_kernels[4];
//	QCLKernel* d_kernel;
//};

class oclBayerFilter
{
public:
	oclBayerFilter(QCLContext* ctx)
		: d_ctx(ctx)
		, d_kernel(nullptr)
	{
		QCLProgram program = ctx->createProgramFromSourceFile("kernels/2d/bayer.cl");
		if(program.isNull() ||
		   program.build(QList<QCLDevice>(), "-Ikernels/2d/"))
		{
			// I wyciagnij z niego kernele
			d_kernels[cvu::BC_RedGreen  - 1] = program.createKernel
				("convert_rg2gray");
			d_kernels[cvu::BC_GreenRed  - 1] = program.createKernel
				("convert_gr2gray");
			d_kernels[cvu::BC_BlueGreen - 1] = program.createKernel
				("convert_bg2gray");
			d_kernels[cvu::BC_GreenBlue - 1] = program.createKernel
				("convert_gb2gray");
		}
	}

	void setSourceImage(
		const QCLImage2D& src)
	{
		this->src = &src;
	}

	QCLEvent run()
	{
		if(!src)
			return QCLEvent();

		if(!d_kernel)
		{
			dst = *src;
			return QCLEvent();
		}

		QSize dstSize(src->width(), src->height());

		dst = d_ctx->createImage2DDevice
				(morphImageFormat(), dstSize, QCLMemoryObject::WriteOnly);

		QCLWorkSize local(8, 8);
		QCLWorkSize global(src->width()-2, src->height()-2);
		QCLWorkSize offset(1, 1);

		global = global.roundTo(local);

		d_kernel->setLocalWorkSize(local);
		d_kernel->setGlobalWorkSize(global);
		d_kernel->setGlobalWorkOffset(offset);

		d_kernel->setArg(0, *src);
		d_kernel->setArg(1, dst);
		QCLEvent evt = d_kernel->run();
		evt.waitForFinished();

		return evt;
	}

	void setBayerFilter(
		cvu::EBayerCode bc)
	{
		if(bc == cvu::BC_None)
			d_kernel = nullptr;
		else
			d_kernel = &d_kernels[bc - 1];
	}

	QCLImage2D outputDeviceImage() const { return dst; }

private:
	QCLContext* d_ctx;

	const QCLImage2D* src;
	QCLImage2D dst;

	QCLKernel d_kernels[4];
	QCLKernel* d_kernel;
};
