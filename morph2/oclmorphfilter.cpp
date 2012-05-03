#include "oclmorphfilter.h"

oclMorphFilter::oclMorphFilter(oclContext* ctx,
	const char* erode, const char* dilate,
	const char* gradient)
	: oclFilter(ctx)
	, morphOp(cvu::MO_None)
{
	// Wczytaj program
	cl::Program program = ctx->createProgram(
		"kernels-buffer2D/morph.cl", "-Ikernels-buffer2D/");

	// I wyciagnij z niego kernele
	kernelErode = ctx->retrieveKernel(program, erode);
	kernelDilate = ctx->retrieveKernel(program, dilate);
	kernelGradient = ctx->retrieveKernel(program, gradient);
	kernelSubtract = ctx->retrieveKernel(program, "subtract");

	structuringElement.size = 0;
}

void oclMorphFilter::setMorphologyOperation(
	cvu::EMorphOperation op)
{
	if (op != cvu::MO_Outline &&
		op != cvu::MO_Skeleton &&
		op != cvu::MO_Skeleton_ZhangSuen)
	{
		morphOp = op;
	}
}

double oclMorphFilter::run()
{
	if(!src)
		return 0.0;

	if (morphOp == cvu::MO_None ||
		structuringElement.size == 0)
	{
		// Passthrough
		if(!ownsOutput)
		{
			// TODO: copy contents of src to dst
		}
		else
		{
			dst = *src;
		}
		return 0.0;
	}

	double elapsed = 0.0;
	prepareDestinationHolder();

	switch(morphOp)
	{
	case cvu::MO_Erode:
		elapsed += runMorphologyKernel(&kernelErode, *src, dst);
		break;
	case cvu::MO_Dilate:
		elapsed += runMorphologyKernel(&kernelDilate, *src, dst);
		break;
	case cvu::MO_Open:
		{
			auto tmp = ctx->createDeviceImage(src->width,
				src->height, ReadWrite);
			elapsed += runMorphologyKernel(&kernelErode, *src, tmp);
			elapsed += runMorphologyKernel(&kernelDilate, tmp, dst);
		}
		break;
	case cvu::MO_Close:
		{
			auto tmp = ctx->createDeviceImage(src->width,
				src->height, ReadWrite);
			elapsed += runMorphologyKernel(&kernelDilate, *src, tmp);
			elapsed += runMorphologyKernel(&kernelErode, tmp, dst);
		}
		break;
	case cvu::MO_Gradient:
		elapsed += runMorphologyKernel(&kernelGradient, *src, dst);
		break;
	case cvu::MO_TopHat:
		{
			auto tmp1 = ctx->createDeviceImage(src->width,
				src->height, ReadWrite);
			auto tmp2 = ctx->createDeviceImage(src->width,
				src->height, ReadWrite);

			elapsed += runMorphologyKernel(&kernelErode, *src, tmp1);
			elapsed += runMorphologyKernel(&kernelDilate, tmp1, tmp2);
			elapsed += runSubtractKernel(*src, tmp2, dst);
		}
		break;
	case cvu::MO_BlackHat:
		{
			auto tmp1 = ctx->createDeviceImage(src->width,
				src->height, ReadWrite);
			auto tmp2 = ctx->createDeviceImage(src->width,
				src->height, ReadWrite);

			elapsed += runMorphologyKernel(&kernelDilate, *src, tmp1);
			elapsed += runMorphologyKernel(&kernelErode, tmp1, tmp2);
			elapsed += runSubtractKernel(tmp2, *src, dst);
		}
		break;
	default: break;
	}

	finishUpDestinationHolder();

	return elapsed;
}

void oclMorphFilter::setStructuringElement(
	const cv::Mat& selement)
{
	std::vector<cl_int2> coords;

	seRadiusX = (selement.cols - 1) / 2;
	seRadiusY = (selement.rows - 1) / 2;

	// Dla implementacji z wykorzystaniem obrazow musimy przesunac uklad wspolrzednych
	// elementu strukturalnego
	bool shiftCoords = true; //(dynamic_cast<MorphOpenCLImage*>(this)) != nullptr;

	// Przetworz wstepnie element strukturalny
	for(int y = 0; y < selement.rows; ++y)
	{
		const uchar* krow = selement.data + selement.step*y;

		for(int x = 0; x < selement.cols; ++x)
		{
			if(krow[x] == 0)
				continue;

			cl_int2 c = {{x, y}};
			if(shiftCoords)
			{
				c.s[0] -= seRadiusX;
				c.s[1] -= seRadiusY;
			}

			coords.push_back(c);
		}
	}

	int csize = static_cast<int>(coords.size());
	printf("Structuring element size (number of 'white' pixels): %d (%dx%d) - %lu B\n",
		csize, 2*seRadiusX+1, 2*seRadiusY+1, sizeof(cl_int2) * csize);

	cl_ulong limit = ctx->deviceDescription().maxConstantBufferSize;
	size_t bmuSize = sizeof(cl_int2) * csize;

	if(bmuSize > limit)
	{
		static char tmpBuf[256];
		snprintf(tmpBuf, sizeof(tmpBuf), "Structuring element is too big:"
			"%lu B out of available %lu B.", bmuSize, limit);
		oclContext::oclError(tmpBuf, CL_OUT_OF_RESOURCES);
	}
	else
	{
		structuringElement = ctx->copyDataToDevice(coords.data(), bmuSize, ReadOnly);
		printf("Transfering structuring element to device took %.5lf ms\n",
			ctx->oclElapsedEvent(structuringElement.evt));
	}
}

double oclMorphFilter::runMorphologyKernel(
	cl::Kernel* kernel,
	const oclImage2DHolder& source,
	oclImage2DHolder& output)
{
	cl_int err;
	err  = kernel->setArg(0, source.img);
	err |= kernel->setArg(1, output.img);
	err |= kernel->setArg(2, structuringElement.buf);
	err |= kernel->setArg(3, structuringElement.size / static_cast<int>(sizeof(cl_int2)));

	if(!oclContext::oclError("Error while setting kernel arguments", err))
		return 0.0;

	cl::NDRange offset(computeOffset(0, 0));
	cl::NDRange gridDim(computeGlobal(0, 0));

	cl::Event evt;
	err = ctx->commandQueue().enqueueNDRangeKernel(
		*kernel, offset, gridDim, ctx->workgroupSize(),
		nullptr, &evt);
	evt.wait();

	oclContext::oclError("Error while executing kernel over ND range!", err);

	return oclContext::oclElapsedEvent(evt);
}

double oclMorphFilter::runSubtractKernel(
	const oclImage2DHolder& sourceA,
	const oclImage2DHolder& sourceB,
	oclImage2DHolder& output)
{
	cl_int err;
	err  = kernelSubtract.setArg(0, sourceA.img);
	err  = kernelSubtract.setArg(1, sourceB.img);
	err |= kernelSubtract.setArg(2, output.img);

	if(!oclContext::oclError("Error while setting kernel arguments", err))
		return 0.0;

	cl::NDRange offset(computeOffset(0, 0));
	cl::NDRange gridDim(computeGlobal(0, 0));

	cl::Event evt;
	err = ctx->commandQueue().enqueueNDRangeKernel(
		kernelSubtract, offset, gridDim, ctx->workgroupSize(),
		nullptr, &evt);
	evt.wait();

	oclContext::oclError("Error while executing kernel over ND range!", err);

	return oclContext::oclElapsedEvent(evt);
}
