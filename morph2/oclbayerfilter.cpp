#include "oclbayerfilter.h"
#include "oclutils.h"

oclBayerFilter::oclBayerFilter(QCLContext* ctx)
	: oclFilter(ctx)
	, d_kernel(nullptr)
{
	printf("\n*---- Bayer filter initialization ----*\n");

	// Wczytaj program
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

qreal oclBayerFilter::run()
{
	if(!d_src)
		return 0.0;

	if(!d_kernel)
	{
		d_dst = *d_src;
		return 0.0;
	}

	prepareDestinationHolder();

	QCLWorkSize offset(computeOffset(1, 1));
	QCLWorkSize global(computeGlobal(1, 1));

	d_kernel->setLocalWorkSize(d_localSize);
	d_kernel->setGlobalWorkSize(global);
	d_kernel->setGlobalWorkOffset(offset);

	d_kernel->setArg(0, *d_src);
	d_kernel->setArg(1,  d_dst);
	QCLEvent evt = d_kernel->run();
	evt.waitForFinished();

	finishUpDestinationHolder();

	return oclUtils::eventDuration(evt);
}

void oclBayerFilter::setBayerFilter(cvu::EBayerCode bc)
{
	if(bc == cvu::BC_None)
		d_kernel = nullptr;
	else
		d_kernel = &d_kernels[bc - 1];
}
