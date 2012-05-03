#include "oclmorphhitmissfilter.h"

#ifdef _MSC_VER
	#define snprintf _snprintf
#endif

oclMorphHitMissFilter::oclMorphHitMissFilter(
	oclContext* ctx, bool atomicCounters)
	: oclFilter(ctx)
	, hmOp(cvu::MO_None)
{
	std::string opts = "-Ikernels-buffer2D/";
	if(atomicCounters)
	{
		opts += " -DUSE_ATOMIC_COUNTERS";
		printf("Using atomic counters instead of global atomic operations\n");
	}

	// Wczytaj program
	cl::Program program = ctx->createProgram(
		"kernels-buffer2D/hitmiss.cl", opts.c_str());

	// I wyciagnij z niego kernele
	kernelOutline = ctx->retrieveKernel(program, "outline");
	kernelSkeleton_pass[0]  = ctx->retrieveKernel(program, "skeletonZhang_pass1");
	kernelSkeleton_pass[1]  = ctx->retrieveKernel(program, "skeletonZhang_pass2");

	for(int i = 0; i < 8; ++i)
	{
		char kernelName[128];
		snprintf(kernelName, sizeof(kernelName), "skeleton_iter%d", i+1);
		kernelSkeleton_iter[i] = ctx->retrieveKernel(program, kernelName);
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
	if(!src)
		return 0.0;

	if(hmOp == cvu::MO_None)
	{
		// Passthrough
		if(!ownsOutput)
		{

		}
		else
		{
			dst = *src;
		}		
		return 0.0;
	}

	double elapsed = 0.0;
	int iters = 1;

	prepareDestinationHolder();

	// Kazda z operacji hit-miss wymaga by wyjscie bylo rowne zrodlu na poczatku
	// - jest tak poniewaz HM zapisuje tylko te piksele ktore modyfikuje
	ctx->copyDeviceImage(*src, dst);
	elapsed += oclContext::oclElapsedEvent(dst.evt);

	switch(hmOp)
	{
	case cvu::MO_Outline:
		elapsed += runHitMissKernel(&kernelOutline, *src, dst);
		break;
	case cvu::MO_Skeleton:
		{
			iters = 0;
			auto tmp = ctx->createDeviceImage(
				src->width, src->height, ReadWrite);

			// Skopiuj obraz zrodlowy do dodatkowego tymczasowego
			ctx->copyDeviceImage(*src, tmp);
			elapsed += oclContext::oclElapsedEvent(tmp.evt);

			// Licznik atomowy (ew. zwyczajny bufor)
			static cl_uint d_init = 0;
			auto atomicCounter = ctx->copyDataToDevice(&d_init, sizeof(cl_uint), ReadWrite);
			elapsed += oclContext::oclElapsedEvent(atomicCounter.evt);

			do
			{
				iters++;

				// 8 operacji hit miss, 2 elementy strukturalnego, 4 orientacje
				for(int i = 0; i < 8; ++i)
				{
					elapsed += runHitMissKernel(&kernelSkeleton_iter[i],
						tmp, dst, &atomicCounter);

					// Kopiowanie obrazu
					ctx->copyDeviceImage(dst, tmp);
					elapsed += oclContext::oclElapsedEvent(tmp.evt);
				}

				// Sprawdz ile pikseli zostalo zmodyfikowanych
				cl_uint diff;
				ctx->readAtomicCounter<cl_uint>(atomicCounter, &diff);
				elapsed += oclContext::oclElapsedEvent(atomicCounter.evt);

				printf("Iteration: %3d, pixel changed: %5d\r", iters, diff);

				// Sprawdz warunek stopu
				if(diff == 0)
					break;

				ctx->zeroAtomicCounter<cl_uint>(atomicCounter);
				elapsed += oclContext::oclElapsedEvent(atomicCounter.evt);

			} while (true);

			printf("\n");
		}
		break;
	case cvu::MO_Skeleton_ZhangSuen:
		{
			iters = 0;

			// Potrzebowac bedziemy dodatkowego bufora tymczasowego
			auto tmp = ctx->createDeviceImage(
				src->width, src->height, ReadWrite);

			// Skopiuj obraz zrodlowy do wyjsciowego
			ctx->copyDeviceImage(*src, dst);
			elapsed += oclContext::oclElapsedEvent(dst.evt);

			// Licznik atomowy (ew. zwyczajny bufor)
			static cl_uint d_init = 0;
			auto atomicCounter = ctx->copyDataToDevice(&d_init, sizeof(cl_uint), ReadWrite);
			elapsed += oclContext::oclElapsedEvent(atomicCounter.evt);

			// Tablica LUT dla operacji szkieletyzacji Zhanga-Suena
			auto zhLut = ctx->copyDataToDevice(cvu::skeletonZHLutTable,
				sizeof(cvu::skeletonZHLutTable), ReadOnly);
			elapsed += oclContext::oclElapsedEvent(zhLut.evt);

			do
			{
				iters++;

				// odd pass
				ctx->copyDeviceImage(dst, tmp);
				elapsed += oclContext::oclElapsedEvent(tmp.evt);
				elapsed += runHitMissKernel(&kernelSkeleton_pass[0],
					tmp, dst, &zhLut, &atomicCounter);

				// even pass
				ctx->copyDeviceImage(dst, tmp);
				elapsed += oclContext::oclElapsedEvent(tmp.evt);
				elapsed += runHitMissKernel(&kernelSkeleton_pass[1],
					tmp, dst, &zhLut, &atomicCounter);

				// Sprawdz ile pikseli zostalo zmodyfikowanych
				cl_uint diff;
				ctx->readAtomicCounter<cl_uint>(atomicCounter, &diff);
				elapsed += oclContext::oclElapsedEvent(atomicCounter.evt);

				printf("Iteration: %3d, pixel changed: %5d\r", iters, diff);

				// Sprawdz warunek stopu
				if(diff == 0)
					break;

				ctx->zeroAtomicCounter<cl_uint>(atomicCounter);
				elapsed += oclContext::oclElapsedEvent(atomicCounter.evt);

			} while (true);

			printf("\n");
		}
		break;
	default: break;
	}

	finishUpDestinationHolder();
	return elapsed;
}

double oclMorphHitMissFilter::runHitMissKernel(
	cl::Kernel* kernel,
	const oclImage2DHolder& source, oclImage2DHolder& output,
	const oclBufferHolder* lut,
	oclBufferHolder* atomicCounter)
{
	cl_int err;
	err  = kernel->setArg(0, source.img);
	err |= kernel->setArg(1, output.img);

	if(lut) err |= kernel->setArg(2, lut->buf);
	if(atomicCounter && lut) err |= kernel->setArg(3, atomicCounter->buf);
	else if (atomicCounter) err |= kernel->setArg(2, atomicCounter->buf);

	if(!oclContext::oclError("Error while setting kernel arguments", err))
		return 0.0;

	cl::NDRange offset(computeOffset(1, 1));
	cl::NDRange gridDim(computeGlobal(1, 1));

	cl::Event evt;
	err = ctx->commandQueue().enqueueNDRangeKernel(
		*kernel, offset, gridDim, ctx->workgroupSize(),
		nullptr, &evt);
	evt.wait();

	oclContext::oclError("Error while executing kernel over ND range!", err);

	return oclContext::oclElapsedEvent(evt);
}
