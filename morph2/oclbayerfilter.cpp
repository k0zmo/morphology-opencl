#include "oclbayerfilter.h"

oclBayerFilter::oclBayerFilter(
	oclContext* ctx)
	: oclFilter(ctx)
	, kernel(nullptr)
{
	// Wczytaj program
	cl::Program programBayer = ctx->createProgram(
		"kernels-buffer2D/bayer.cl", "-Ikernels-buffer2D/ -DGRAYSCALE");

	// I wyciagnij z niego kernele
	kernels[cvu::BC_RedGreen  - 1] = ctx->retrieveKernel(
		programBayer, "convert_rg2gray");
	kernels[cvu::BC_GreenRed  - 1] = ctx->retrieveKernel(
		programBayer, "convert_gr2gray");
	kernels[cvu::BC_BlueGreen - 1] = ctx->retrieveKernel(
		programBayer, "convert_bg2gray");
	kernels[cvu::BC_GreenBlue - 1] = ctx->retrieveKernel(
		programBayer, "convert_gb2gray");
}

double oclBayerFilter::run()
{
	if(!src)
		return 0.0;

	if(!kernel)
	{
		// Passthrough
		if(!ownsOutput)
		{
			// TODO
		}
		else
		{
			dst = *src;
		}		
		return 0.0;
	}

	prepareDestinationHolder();

	cl::NDRange offset(computeOffset(1, 1));
	cl::NDRange gridDim(computeGlobal(1, 1));

#if 0
	// Zdaje sie ze jest nowy interfejs c++ dla uruchamiania kerneli
	// Taki bardziej funkcyjny (funktorowy?)
	cl_int err = CL_SUCCESS;
	cl::make_kernel<cl::Image2D, cl::Image2D> kernelFunc(*kernel, &err);
	cl::Event evt = kernelFunc(
		cl::EnqueueArgs(ctx->commandQueue(), offset, gridDim, blockDim),
		src->img, dst.img);
#else

	cl_int err;
	err  = kernel->setArg(0, src->img);
	err |= kernel->setArg(1, dst.img);

	if(!oclContext::oclError("Error while setting kernel arguments", err))
		return 0.0;

	cl::Event evt;
	err = ctx->commandQueue().enqueueNDRangeKernel(
		*kernel, offset, gridDim, ctx->workgroupSize(),
		nullptr, &evt);
#endif
	evt.wait();

	oclContext::oclError("Error while executing kernel over ND range!", err);
		
	finishUpDestinationHolder();
	return oclContext::oclElapsedEvent(evt);
}

void oclBayerFilter::setBayerFilter(
	cvu::EBayerCode bc)
{
	if(bc == cvu::BC_None)
		kernel = nullptr;
	else
		kernel = &kernels[bc - 1];
}
