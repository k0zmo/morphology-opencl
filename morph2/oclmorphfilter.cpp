#include "oclmorphfilter.h"
#include "oclutils.h"

oclMorphFilter::oclMorphFilter(QCLContext* ctx,
	const char* erode, const char* dilate,
	const char* gradient)
	: oclFilter(ctx)
	, morphOp(cvu::MO_None)
{
	printf("\n*---- Morphology filter initialization ----*\n");

	// Wczytaj program
	QCLProgram program = ctx->createProgramFromSourceFile("kernels/2d/morph.cl");
	if(program.isNull() ||
		program.build(QList<QCLDevice>(), "-Ikernels/2d/"))
	{
		// I wyciagnij z niego kernele
		kernelErode = program.createKernel(erode);
		kernelDilate = program.createKernel(dilate);
		kernelGradient = program.createKernel(gradient);
		kernelSubtract = program.createKernel("subtract");
	}
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

qreal oclMorphFilter::run()
{
	if(!d_src)
		return 0.0;

	// Passthrough
	if(morphOp == cvu::MO_None ||
	   structuringElement.isNull())
	{
		// TODO
		//if(!ownsOutput)
		d_dst = *d_src;
		return 0.0;
	}

	qreal elapsed = 0.0;
	prepareDestinationHolder();

	switch(morphOp)
	{
	case cvu::MO_Erode:
		elapsed += runMorphologyKernel(&kernelErode, *d_src, d_dst);
		break;
	case cvu::MO_Dilate:
		elapsed += runMorphologyKernel(&kernelDilate, *d_src, d_dst);
		break;
	case cvu::MO_Open:
		{
			auto tmp = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), 
				QSize(d_src->width(), d_src->height()), 
				QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelErode, *d_src, tmp);
			elapsed += runMorphologyKernel(&kernelDilate, tmp, d_dst);
		}
		break;
	case cvu::MO_Close:
		{
			auto tmp = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), 
				 QSize(d_src->width(), d_src->height()), 
				 QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelDilate, *d_src, tmp);
			elapsed += runMorphologyKernel(&kernelErode, tmp, d_dst);
		}
		break;
	case cvu::MO_Gradient:
		elapsed += runMorphologyKernel(&kernelGradient, *d_src, d_dst);
		break;
	case cvu::MO_TopHat:
		{
			QSize tmpSize(d_src->width(), d_src->height());

			auto tmp1 = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), tmpSize, QCLMemoryObject::ReadWrite);
			auto tmp2 = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), tmpSize, QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelErode, *d_src, tmp1);
			elapsed += runMorphologyKernel(&kernelDilate, tmp1, tmp2);
			elapsed += runSubtractKernel(*d_src, tmp2, d_dst);
		}
		break;
	case cvu::MO_BlackHat:
		{
			QSize tmpSize(d_src->width(), d_src->height());

			auto tmp1 = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), tmpSize, QCLMemoryObject::ReadWrite);
			auto tmp2 = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), tmpSize, QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelDilate, *d_src, tmp1);
			elapsed += runMorphologyKernel(&kernelErode, tmp1, tmp2);
			elapsed += runSubtractKernel(tmp2, *d_src, d_dst);
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

	size_t bmuSize = sizeof(cl_int2) * csize;
	cl_ulong limit = d_ctx->defaultDevice().maximumConstantBufferSize();
	
	if(bmuSize > limit)
	{
		printf("Structuring element is too big:"
			"%lu B out of available %lu B.", bmuSize, limit);
		//static char tmpBuf[256];
		//snprintf(tmpBuf, sizeof(tmpBuf), "Structuring element is too big:"
		//	"%lu B out of available %lu B.", bmuSize, limit);
		// TODO:
		//oclContext::oclError(tmpBuf, CL_OUT_OF_RESOURCES);
	}
	else
	{
		structuringElement = d_ctx->createBufferDevice
			(bmuSize, QCLMemoryObject::ReadOnly);
		QCLEvent evt = structuringElement.writeAsync(0, coords.data(), bmuSize);
		evt.waitForFinished();

		printf("Transfering structuring element to device took %.5lf ms\n",
			(evt.finishTime() - evt.runTime()) / 1000000.0f);
	}
}

qreal oclMorphFilter::runMorphologyKernel(
	QCLKernel* kernel,
	const QCLImage2D& source,
	QCLImage2D& output)
{
	kernel->setArg(0, source);
	kernel->setArg(1, output);
	kernel->setArg(2, structuringElement);
	kernel->setArg(3, structuringElement.size() / sizeof(cl_int2));

	kernel->setLocalWorkSize(localWorkSize());
	kernel->setGlobalWorkOffset(computeOffset(0, 0));
	kernel->setGlobalWorkSize(computeGlobal(0, 0));

	QCLEvent evt = kernel->run();
	evt.waitForFinished();

	return oclUtils::eventDuration(evt);
}

qreal oclMorphFilter::runSubtractKernel(
	const QCLImage2D& sourceA,
	const QCLImage2D& sourceB,
	QCLImage2D& output)
{
	kernelSubtract.setArg(0, sourceA);
	kernelSubtract.setArg(1, sourceB);
	kernelSubtract.setArg(2, output);

	kernelSubtract.setLocalWorkSize(localWorkSize());
	kernelSubtract.setGlobalWorkOffset(computeOffset(0, 0));
	kernelSubtract.setGlobalWorkSize(computeGlobal(0, 0));

	QCLEvent evt = kernelSubtract.run();
	evt.waitForFinished();

	return oclUtils::eventDuration(evt);
}
