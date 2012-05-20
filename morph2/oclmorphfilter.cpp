#include "oclmorphfilter.h"
#include "oclutils.h"

oclMorphFilter::oclMorphFilter(QCLContext* ctx,
	const char* erode, const char* dilate,
	const char* gradient)
	: oclFilter(ctx)
	, seRadiusX(0)
	, seRadiusY(0)
	, csize(0)
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

void oclMorphFilter::setStructuringElement(const cv::Mat& selement)
{
	structuringElement = oclUtils::setStructuringElement
		(d_ctx, selement, true, seRadiusX, seRadiusY);
	csize = structuringElement.size() / sizeof(cl_int2);
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
				 d_size, QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelErode, *d_src, tmp);
			elapsed += runMorphologyKernel(&kernelDilate, tmp, d_dst);
		}
		break;
	case cvu::MO_Close:
		{
			auto tmp = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), 
				 d_size, QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelDilate, *d_src, tmp);
			elapsed += runMorphologyKernel(&kernelErode, tmp, d_dst);
		}
		break;
	case cvu::MO_Gradient:
		elapsed += runMorphologyKernel(&kernelGradient, *d_src, d_dst);
		break;
	case cvu::MO_TopHat:
		{
			auto tmp1 = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), d_size, QCLMemoryObject::ReadWrite);
			auto tmp2 = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), d_size, QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelErode, *d_src, tmp1);
			elapsed += runMorphologyKernel(&kernelDilate, tmp1, tmp2);
			elapsed += runSubtractKernel(*d_src, tmp2, d_dst);
		}
		break;
	case cvu::MO_BlackHat:
		{
			auto tmp1 = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), d_size, QCLMemoryObject::ReadWrite);
			auto tmp2 = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), d_size, QCLMemoryObject::ReadWrite);

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

qreal oclMorphFilter::runMorphologyKernel(
	QCLKernel* kernel,
	const QCLImage2D& source,
	QCLImage2D& output)
{
	kernel->setArg(0, source);
	kernel->setArg(1, output);
	kernel->setArg(2, structuringElement);
	kernel->setArg(3, csize);

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

// ____________________________________________________________________________

oclMorphFilterBuffer::oclMorphFilterBuffer(QCLContext* ctx,
	const char *erode, const char *dilate,
	const char *gradient, const char* subtract)
	: oclFilterBuffer(ctx)
	, seRadiusX(0)
	, seRadiusY(0)
	, csize(0)
	, morphOp(cvu::MO_None)
{
	printf("\n*---- Morphology filter initialization ----*\n");

	QString opts = "-Ikernels/1d/";
	if(1)
	{
		opts += " -DUSE_UCHAR";
		printf("Using uchar/uchar4\n");
	}

	// Wczytaj program
	QCLProgram program = ctx->createProgramFromSourceFile("kernels/1d/morph.cl");
	if(program.isNull() ||
		program.build(QList<QCLDevice>(), opts))
	{
		// I wyciagnij z niego kernele
		kernelErode = program.createKernel(erode);
		kernelDilate = program.createKernel(dilate);
		kernelGradient = program.createKernel(gradient);
		kernelSubtract = program.createKernel(subtract);
	}
}

void oclMorphFilterBuffer::setStructuringElement(const cv::Mat& selement)
{
	structuringElement = oclUtils::setStructuringElement
		(d_ctx, selement, false, seRadiusX, seRadiusY);
	csize = structuringElement.size() / sizeof(cl_int2);
}

void oclMorphFilterBuffer::setMorphologyOperation(cvu::EMorphOperation op)
{
	if (op != cvu::MO_Outline &&
		op != cvu::MO_Skeleton &&
		op != cvu::MO_Skeleton_ZhangSuen)
	{
		morphOp = op;
	}
}

qreal oclMorphFilterBuffer::run()
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
			size_t bufferSize = d_size.width() * d_size.height();
			auto tmp = d_ctx->createBufferDevice
				(bufferSize, QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelErode, *d_src, tmp);
			elapsed += runMorphologyKernel(&kernelDilate, tmp, d_dst);
		}
		break;
	case cvu::MO_Close:
		{
			size_t bufferSize = d_size.width() * d_size.height();
			auto tmp = d_ctx->createBufferDevice
				(bufferSize, QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelDilate, *d_src, tmp);
			elapsed += runMorphologyKernel(&kernelErode, tmp, d_dst);
		}
		break;
	case cvu::MO_Gradient:
		elapsed += runMorphologyKernel(&kernelGradient, *d_src, d_dst);
		break;
	case cvu::MO_TopHat:
		{
			size_t bufferSize = d_size.width() * d_size.height();

			auto tmp1 = d_ctx->createBufferDevice
				(bufferSize, QCLMemoryObject::ReadWrite);
			auto tmp2 = d_ctx->createBufferDevice
				(bufferSize, QCLMemoryObject::ReadWrite);

			elapsed += runMorphologyKernel(&kernelErode, *d_src, tmp1);
			elapsed += runMorphologyKernel(&kernelDilate, tmp1, tmp2);
			elapsed += runSubtractKernel(*d_src, tmp2, d_dst);
		}
		break;
	case cvu::MO_BlackHat:
		{
			size_t bufferSize = d_size.width() * d_size.height();

			auto tmp1 = d_ctx->createBufferDevice
				(bufferSize, QCLMemoryObject::ReadWrite);
			auto tmp2 = d_ctx->createBufferDevice
				(bufferSize, QCLMemoryObject::ReadWrite);

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

qreal oclMorphFilterBuffer::runMorphologyKernel(QCLKernel* kernel,
	const QCLBuffer& source, QCLBuffer& output)
{
	bool useLocal = kernel->name().contains("_local");

	cl_int4 seSize = { seRadiusX, seRadiusY, csize, 0 };
	cl_int2 imageSize = { d_size.width(), d_size.height() };

	int apronX = seRadiusX * 2;
	int apronY = seRadiusY * 2;

	kernel->setArg(0, source);
	kernel->setArg(1, output);
	kernel->setArg(2, structuringElement);
	kernel->setArg(3, &seSize, sizeof(cl_int4));
	kernel->setArg(4, &imageSize, sizeof(cl_int2));

	kernel->setLocalWorkSize(localWorkSize());
	kernel->setGlobalWorkOffset(computeOffset(0, 0));

	QCLWorkSize global(computeGlobal(0, 0));
	global = QCLWorkSize(global.width() - apronX, global.height() - apronY);
	kernel->setGlobalWorkSize(global.roundTo(d_localSize));

	// Trzeba ustawic dodatkowe argumenty kernela
	if(useLocal)
	{
		cl_int2 sharedSize = {
			oclUtils::roundUp(d_localSize.width() + apronX, 4),
			d_localSize.height() + apronY
		};
		size_t sharedBlockSize = sharedSize.s[0] * sharedSize.s[1];

		printf("LDS usage (%dx%d): %d B\n", 
			sharedSize.s[0], sharedSize.s[1], sharedBlockSize);

		kernel->setArg(5, nullptr, sharedBlockSize);
		kernel->setArg(6, &sharedSize, sizeof(cl_int2));
	}

	QCLEvent evt = kernel->run();
	evt.waitForFinished();

	return oclUtils::eventDuration(evt);
}

qreal oclMorphFilterBuffer::runSubtractKernel(const QCLBuffer& sourceA, 
	const QCLBuffer& sourceB, QCLBuffer& output)
{
	bool sub4 = kernelSubtract.name().contains("4");
	int xitems = d_size.width();
	if(sub4) xitems /= 4;

	kernelSubtract.setLocalWorkSize(localWorkSize());
	kernelSubtract.setGlobalWorkOffset(QCLWorkSize(0, 0));
	kernelSubtract.setGlobalWorkSize
		(QCLWorkSize(xitems * d_size.height())
			.roundTo(localWorkSize()));

	kernelSubtract.setArg(0, sourceA);
	kernelSubtract.setArg(1, sourceB);
	kernelSubtract.setArg(2, output);
	kernelSubtract.setArg(3, xitems * d_size.height());

	QCLEvent evt = kernelSubtract.run();
	evt.waitForFinished();

	return oclUtils::eventDuration(evt);
}