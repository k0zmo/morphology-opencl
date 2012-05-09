#include "oclmorphhitmissfilter.h"
#include "oclutils.h"

#ifdef _MSC_VER
	#define snprintf _snprintf
#endif

oclMorphHitMissFilter::oclMorphHitMissFilter(
	QCLContext* ctx, bool atomicCounters)
	: oclFilter(ctx)
	, hmOp(cvu::MO_None)
{
	QString opts = "-Ikernels/2d/";
	if(atomicCounters)
	{
		opts += " -DUSE_ATOMIC_COUNTERS";
		printf("Using atomic counters instead of global atomic operations\n");
	}

	// Wczytaj program
	QCLProgram program = ctx->createProgramFromSourceFile("kernels/2d/hitmiss.cl");
	if(program.isNull() ||
		program.build(QList<QCLDevice>(), "-Ikernels/2d/"))
	{
		// I wyciagnij z niego kernele
		kernelOutline = program.createKernel("outline");
		kernelSkeleton_pass[0] = program.createKernel("skeletonZhang_pass1");
		kernelSkeleton_pass[1] = program.createKernel("skeletonZhang_pass2");

		for(int i = 0; i < 8; ++i)
		{
			QString kernelName = QString("skeleton_iter") + QString::number(i+1);
			kernelSkeleton_iter[i] = program.createKernel(kernelName);
		}
	}
}

void oclMorphHitMissFilter::setHitMissOperation(
	cvu::EMorphOperation op)
{
	if (op == cvu::MO_Outline ||
		op == cvu::MO_Skeleton ||
		op == cvu::MO_Skeleton_ZhangSuen)
	{
		hmOp = op;
	}
}

double oclMorphHitMissFilter::run()
{
	if(!d_src)
		return 0.0;

	if(hmOp == cvu::MO_None)
	{
		// Passthrough
		//if(!ownsOutput)
		d_dst = *d_src;	
		return 0.0;
	}

	double elapsed = 0.0;
	int iters = 1;

	prepareDestinationHolder();

	// Kazda z operacji hit-miss wymaga by wyjscie bylo rowne zrodlu na poczatku
	// - jest tak poniewaz HM zapisuje tylko te piksele ktore modyfikuje
	elapsed += copyImage2D(*d_src, d_dst);

	switch(hmOp)
	{
	case cvu::MO_Outline:
		elapsed += runHitMissKernel(&kernelOutline, *d_src, d_dst);
		break;
	case cvu::MO_Skeleton:
		{
			iters = 0;

			// Potrzebowac bedziemy dodatkowego bufora tymczasowego
			auto tmp = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), 
				 QSize(d_src->width(), d_src->height()),
				 QCLMemoryObject::ReadWrite);

			// Skopiuj obraz zrodlowy do dodatkowego tymczasowego
			elapsed += copyImage2D(*d_src, tmp);

			// Licznik atomowy (ew. zwyczajny bufor)
			QCLBuffer atomicCounter = d_ctx->createBufferDevice
				(sizeof(cl_uint), QCLMemoryObject::ReadWrite);
			elapsed += zeroAtomicCounter(atomicCounter);

			do
			{
				iters++;

				// 8 operacji hit miss, 2 elementy strukturalnego, 4 orientacje
				for(int i = 0; i < 8; ++i)
				{
					elapsed += runHitMissKernel(&kernelSkeleton_iter[i],
						tmp, d_dst, &atomicCounter);

					// Kopiowanie obrazu
					elapsed += copyImage2D(d_dst, tmp);
				}

				// Sprawdz ile pikseli zostalo zmodyfikowanych
				cl_uint diff;
				elapsed += readAtomicCounter(atomicCounter, diff);

				printf("Iteration: %3d, pixel changed: %5d\r", iters, diff);

				// Sprawdz warunek stopu
				if(diff == 0)
					break;

				elapsed += zeroAtomicCounter(atomicCounter);
			} while (true);

			printf("\n");
		}
		break;
	case cvu::MO_Skeleton_ZhangSuen:
		{
			iters = 0;

			// Potrzebowac bedziemy dodatkowego bufora tymczasowego
			auto tmp = d_ctx->createImage2DDevice
				(oclUtils::morphImageFormat(), 
				QSize(d_src->width(), d_src->height()),
				QCLMemoryObject::ReadWrite);

			// Skopiuj obraz zrodlowy do dodatkowego tymczasowego
			elapsed += copyImage2D(*d_src, tmp);

			// Licznik atomowy (ew. zwyczajny bufor)
			static cl_uint d_init = 0;
			QCLBuffer atomicCounter = d_ctx->createBufferDevice
				(sizeof(cl_uint), QCLMemoryObject::ReadWrite);
			elapsed += zeroAtomicCounter(atomicCounter);

			// Tablica LUT dla operacji szkieletyzacji Zhanga-Suena
			QCLBuffer zhLut = d_ctx->createBufferDevice
				(sizeof(cvu::skeletonZHLutTable), QCLMemoryObject::ReadWrite);
			QCLEvent evt = zhLut.writeAsync
				(0, cvu::skeletonZHLutTable, 
				sizeof(cvu::skeletonZHLutTable));
			evt.waitForFinished();
			elapsed += oclUtils::eventDuration(evt);

			do
			{
				iters++;

				// odd pass
				elapsed += copyImage2D(d_dst, tmp);
				elapsed += runHitMissKernel(&kernelSkeleton_pass[1],
					tmp, d_dst, &zhLut, &atomicCounter);

				// even pass
				elapsed += copyImage2D(d_dst, tmp);
				elapsed += runHitMissKernel(&kernelSkeleton_pass[0],
					tmp, d_dst, &zhLut, &atomicCounter);

				// Sprawdz ile pikseli zostalo zmodyfikowanych
				cl_uint diff;
				elapsed += readAtomicCounter(atomicCounter, diff);

				printf("Iteration: %3d, pixel changed: %5d\r", iters, diff);

				// Sprawdz warunek stopu
				if(diff == 0)
					break;

				elapsed += zeroAtomicCounter(atomicCounter);

			} while (true);

			printf("\n");
		}
		break;
	default: break;
	}

	finishUpDestinationHolder();
	return elapsed;
}

qreal oclMorphHitMissFilter::runHitMissKernel(
	QCLKernel* kernel,
	const QCLImage2D& source, QCLImage2D& output,
	const QCLBuffer* lut,
	QCLBuffer* atomicCounter)
{
	kernel->setArg(0, source);
	kernel->setArg(1, output);

	if(lut) kernel->setArg(2, *lut);
	if(atomicCounter && lut) kernel->setArg(3, *atomicCounter);
	else if (atomicCounter) kernel->setArg(2, *atomicCounter);

	kernel->setLocalWorkSize(localWorkSize());
	kernel->setGlobalWorkOffset(computeOffset(1, 1));
	kernel->setGlobalWorkSize(computeGlobal(1, 1));

	QCLEvent evt = kernel->run();
	evt.waitForFinished();

	return oclUtils::eventDuration(evt);
}

qreal oclMorphHitMissFilter::readAtomicCounter(QCLBuffer& buf, cl_uint& dst)
{
	QCLEvent evt = buf.readAsync(0, &dst, sizeof(cl_uint));
	evt.waitForFinished();
	return oclUtils::eventDuration(evt);
}

qreal oclMorphHitMissFilter::zeroAtomicCounter(QCLBuffer& buf)
{
	static int init = 0;
	QCLEvent evt = buf.writeAsync(0, &init, sizeof(cl_uint));
	evt.waitForFinished();
	return oclUtils::eventDuration(evt);
}

qreal oclMorphHitMissFilter::copyImage2D(const QCLImage2D& src, QCLImage2D& dst)
{
	QCLEvent evt = const_cast<QCLImage2D&>(src).copyToAsync
		(QRect(0, 0, src.width(), src.height()),
		dst, QPoint(0, 0));
	evt.waitForFinished();
	return oclUtils::eventDuration(evt);
}