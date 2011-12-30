#include "morphoclbuffer.h"

#include <QSettings>

MorphOpenCLBuffer::MorphOpenCLBuffer()
	: MorphOpenCL()
{
	QSettings settings("./settings.cfg", QSettings::IniFormat);
}
// -------------------------------------------------------------------------
bool MorphOpenCLBuffer::initOpenCL()
{
	MorphOpenCL::initOpenCL();

	QSettings s("./settings.cfg", QSettings::IniFormat);
	QString opts = "-Ikernels-buffer1D/";

	// Typ danych (uchar czy uint)
	if(s.value("opencl/datatype", "0").toInt() == 0)
	{
		useUint = false;
		opts += " -DUSE_UCHAR";
	}
	else
	{
		useUint = true;	
	}

	if(s.value("opencl/atomiccounters", false).toBool())
		opts += " -DUSE_ATOMIC_COUNTERS";

	// do ewentualnej rekompilacji z podaniem innego parametry -D
	erodeParams.programName = "kernels-buffer1D/erode.cl";
	erodeParams.options = opts;
	erodeParams.kernelName = s.value("kernel-buffer1D/erode", "erode").toString();
	erodeParams.needRecompile = erodeParams.kernelName.contains("_pragma", Qt::CaseSensitive);

	dilateParams.programName = "kernels-buffer1D/dilate.cl";
	dilateParams.options = opts;
	dilateParams.kernelName = s.value("kernel-buffer1D/dilate", "dilate").toString();
	dilateParams.needRecompile = dilateParams.kernelName.contains("_pragma", Qt::CaseSensitive);

	gradientParams.programName = "kernels-buffer1D/gradient.cl";
	gradientParams.options = opts;
	gradientParams.kernelName = s.value("kernel-buffer1D/gradient", "gradient").toString();
	gradientParams.needRecompile = gradientParams.kernelName.contains("_pragma", Qt::CaseSensitive);

	// Wczytaj programy (rekompilowalne)
	cl::Program perode = createProgram(erodeParams.programName, opts);
	cl::Program pdilate = createProgram(dilateParams.programName, opts);
	cl::Program pgradient = createProgram(gradientParams.programName, opts);

	// Wczytaj reszte programow (nie ma sensu ich rekompilowac)
	cl::Program poutline = createProgram("kernels-buffer1D/outline.cl", opts);
	cl::Program putils = createProgram("kernels-buffer1D/utils.cl", opts);
	cl::Program pskeleton = createProgram("kernels-buffer1D/skeleton.cl", opts);	
	cl::Program pskeletonz = createProgram("kernels-buffer1D/skeleton_zhang.cl", opts);

	// Stworz kernele (nazwy pobierz z pliku konfiguracyjnego)
	kernelErode = createKernel(perode, erodeParams.kernelName);
	kernelDilate = createKernel(pdilate, dilateParams.kernelName);
	kernelGradient = createKernel(pgradient, gradientParams.kernelName);
	kernelSubtract = createKernel(putils, s.value("kernel-buffer1D/subtract", "subtract").toString());

	// subtract4 (wymaga wyrownania wierszy danych do 4 bajtow) czy subtract
	QString sub = s.value("kernel-buffer1D/subtract", "subtract").toString();
	if(sub.endsWith("4")) sub4 = true;
	else sub4 = false;

	// hitmiss
	QString localHitmissStr = s.value("kernel-buffer1D/hitmiss", "global").toString();
	bool localHitmiss = localHitmissStr.contains("local");

	kernelOutline = createKernel(poutline, (localHitmiss ? "outline4_local" : "outline"));
	
	for(int i = 0; i < 8; ++i)
	{
		if(localHitmiss == false)
		{
			QString kernelName = "skeleton_iter" + QString::number(i+1);
			kernelSkeleton_iter[i] = createKernel(pskeleton, kernelName);
		}
		else
		{
			QString kernelName = "skeleton4_iter" + QString::number(i+1) + "_local";
			kernelSkeleton_iter[i] = createKernel(pskeleton, kernelName);
		}
	}

	if(localHitmiss == false)
	{
		kernelSkeleton_pass[0] = createKernel(pskeletonz, "skeletonZhang_pass1");
		kernelSkeleton_pass[1] = createKernel(pskeletonz, "skeletonZhang_pass2");
	}
	else
	{
		kernelSkeleton_pass[0] = createKernel(pskeletonz, "skeletonZhang4_pass1_local");
		kernelSkeleton_pass[1] = createKernel(pskeletonz, "skeletonZhang4_pass2_local");
	}

	return true;
}
// -------------------------------------------------------------------------
void MorphOpenCLBuffer::setSourceImage(const cv::Mat* newSrc)
{
	cl_int err;
	sourceBuffer.cpu = newSrc;

	sourceBuffer.gpuWidth = roundUp(newSrc->cols, workGroupSizeX);
	sourceBuffer.gpuHeight = roundUp(newSrc->rows, workGroupSizeY);
	sourceBuffer.gpu = cl::Buffer(context, CL_MEM_READ_ONLY,
		bufferSize(), nullptr, &err);
	clError("Error while creating OpenCL source buffer", err);

	void* srcptr = const_cast<uchar*>(newSrc->ptr<uchar>());
	uint* ptr = nullptr;

	// Konwersja uchar -> uint
	if(useUint)
	{
		ptr = new uint[newSrc->cols * newSrc->rows];
		const uchar* uptr = newSrc->ptr<uchar>();
		for(int i = 0; i < newSrc->cols * newSrc->rows; ++i)
			ptr[i] = (int)(uptr[i]);

		srcptr = ptr;
	}

	cl::Event evt;

	// Skopiuj dane tak by byly odpowiednio wyrownane
	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	cl::size_t<3> region;
	region[0] = newSrc->cols;
	region[1] = newSrc->rows;
	region[2] = 1;

	size_t buffer_row_pitch = sourceBuffer.gpuWidth;
	size_t host_row_pitch = newSrc->cols;

	if(useUint)
	{
		region[0] *= sizeof(uint);
		buffer_row_pitch *= sizeof(uint);
		host_row_pitch *= sizeof(uint);
	}

	err = cq.enqueueWriteBufferRect(sourceBuffer.gpu, CL_TRUE, 
		origin, origin, region, 
		buffer_row_pitch, 0, 
		host_row_pitch, 0, 
		srcptr, 0, &evt);

	evt.wait();
	clError("Error while writing new data to OpenCL source buffer!", err);

	// Podaj czas trwania transferu
	cl_ulong delta = elapsedEvent(evt);
	printf("Transfering source image to GPU took %.05lfms\n", delta * 0.000001);

	delete [] ptr;
}
// -------------------------------------------------------------------------
double MorphOpenCLBuffer::morphology(EOperationType opType, cv::Mat& dst, int& iters)
{
	int dstSizeX = sourceBuffer.cpu->cols;
	int dstSizeY = sourceBuffer.cpu->rows;

	iters = 1;
	cl_ulong elapsed = 0;

	// Bufor docelowy
	cl::Buffer clDst = createBuffer(CL_MEM_WRITE_ONLY);

	switch(opType)
	{
	case OT_Erode:
		elapsed += morphologyErode(sourceBuffer.gpu, clDst);
		dstSizeX -= kradiusx*2;
		dstSizeY -= kradiusy*2;
		break;
	case OT_Dilate:
		elapsed += morphologyDilate(sourceBuffer.gpu, clDst);
		dstSizeX -= kradiusx*2;
		dstSizeY -= kradiusy*2;
		break;
	case OT_Open:
		elapsed += morphologyOpen(sourceBuffer.gpu, clDst);
		dstSizeX -= kradiusx*4;
		dstSizeY -= kradiusy*4;
		break;
	case OT_Close:
		elapsed += morphologyClose(sourceBuffer.gpu, clDst);
		dstSizeX -= kradiusx*4;
		dstSizeY -= kradiusy*4;
		break;
	case OT_Gradient:
		elapsed += morphologyGradient(sourceBuffer.gpu, clDst);
		dstSizeX -= kradiusx*2;
		dstSizeY -= kradiusy*2;
		break;
	case OT_TopHat:
		elapsed += morphologyTopHat(sourceBuffer.gpu, clDst);
		dstSizeX -= kradiusx*4;
		dstSizeY -= kradiusy*4;
		break;
	case OT_BlackHat:
		elapsed += morphologyBlackHat(sourceBuffer.gpu, clDst);
		dstSizeX -= kradiusx*4;
		dstSizeY -= kradiusy*4;
		break;
	case OT_Outline:
		elapsed += morphologyOutline(sourceBuffer.gpu, clDst);
		dstSizeX -= 2;
		dstSizeY -= 2;
		break;
	case OT_Skeleton:
		elapsed += morphologySkeleton(sourceBuffer.gpu, clDst, iters);
		dstSizeX -= 2;
		dstSizeY -= 2;
		break;
	case OT_Skeleton_ZhangSuen:
		elapsed += morphologySkeletonZhangSuen(sourceBuffer.gpu, clDst, iters);
		dstSizeX -= 2;
		dstSizeY -= 2;
		break;
	}

	// Zczytaj wynik z karty
	cl_ulong readingTime = readBack(clDst, dst, dstSizeX, dstSizeY);

	double totalTime = (elapsed + readingTime) * 0.000001;
	//FILE* fp = fopen("oo.txt", "a");
	printf("Total time: %.05lf ms (in which %.05lf was a processing time "
		"and %.05lf ms was a transfer time)\n",
		totalTime, elapsed * 0.000001, readingTime * 0.000001);
	//fprintf(fp, "%.05lf\n", elapsed*0.000001);
	//fclose(fp);

	// Ile czasu wszystko zajelo
	return totalTime;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::readBack(const cl::Buffer& source,
	cv::Mat &dst, int dstSizeX, int dstSizeY)
{
	dst = cv::Mat(sourceBuffer.cpu->size(), CV_8U, cv::Scalar(0));
	cl::Event evt;

	cl::size_t<3> origin;
	cl::size_t<3> region;

	origin[0] = (sourceBuffer.cpu->cols - dstSizeX)/2;
	origin[1] = (sourceBuffer.cpu->rows - dstSizeY)/2;
	origin[2] = 0;

	region[0] = dstSizeX;
	region[1] = dstSizeY;
	region[2] = 1;

	size_t buffer_row_pitch = sourceBuffer.gpuWidth;
	size_t host_row_pitch = sourceBuffer.cpu->cols;

	if(useUint)
	{
		// Musimy wczytac wiecej danych
		origin[0] *= sizeof(uint);
		region[0] *= sizeof(uint);

		buffer_row_pitch *= sizeof(uint);
		host_row_pitch *= sizeof(uint);

		// .. do tymczasowego bufora uint'ow
		uint* dstTmp = new uint[sourceBuffer.cpu->size().area()];

		cl_int err = cq.enqueueReadBufferRect(source, CL_FALSE, 
			origin, origin, region, 
			buffer_row_pitch, 0, 
			host_row_pitch, 0, 
			dstTmp, nullptr, &evt);

		clError("Error while reading result to buffer!", err);
		evt.wait();

		// .. a nastepnie zrzutowac do uchar'ow
		uchar* dptr = dst.ptr<uchar>();
		for(int i = 0; i < dst.cols * dst.rows; ++i)
			dptr[i] = static_cast<uchar>(dstTmp[i]);

		delete [] dstTmp;
	}
	else
	{
		cl_int err = cq.enqueueReadBufferRect(source, CL_FALSE, 
			origin, origin, region, 
			buffer_row_pitch, 0, 
			host_row_pitch, 0, 
			dst.ptr<uchar>(), nullptr, &evt);

		clError("Error while reading result to buffer!", err);
		evt.wait();
	}

	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl::Buffer MorphOpenCLBuffer::createBuffer(cl_mem_flags memFlags)
{
	cl_int err;
	cl::Buffer buffer(context,
		memFlags, 
		bufferSize(),
		nullptr, &err);
	clError("Error while creating destination OpenCL buffer!", err);

	return buffer;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::copyBuffer(const cl::Buffer& src, cl::Buffer& dst)
{
	cl::Event evt;
	cq.enqueueCopyBuffer(src, dst, 
		0, 0, 
		bufferSize(), 
		nullptr, &evt);
	evt.wait();
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologyErode(cl::Buffer& src, cl::Buffer& dst)
{
	//printf(" *** erode\n");
	return executeMorphologyKernel(&kernelErode, src, dst);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologyDilate( cl::Buffer& src, cl::Buffer& dst )
{
	//printf(" *** dilate\n");
	return executeMorphologyKernel(&kernelDilate, src, dst);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologyOpen(cl::Buffer& src, cl::Buffer& dst)
{
	// Potrzebowac bedziemy dodatkowego bufora tymczasowego
	cl::Buffer tmpBuffer = createBuffer(CL_MEM_READ_WRITE);

	// dst = dilate(erode(src))
	cl_ulong elapsed = 0;
	//printf(" *** erode\n");
	elapsed += executeMorphologyKernel(&kernelErode, src, tmpBuffer);
	//printf(" *** dilate\n");
	elapsed += executeMorphologyKernel(&kernelDilate, tmpBuffer, dst);

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologyClose(cl::Buffer& src, cl::Buffer& dst)
{
	// Potrzebowac bedziemy dodatkowego bufora tymczasowego
	cl::Buffer tmpBuffer = createBuffer(CL_MEM_READ_WRITE);

	// dst = erode(dilate(src))
	cl_ulong elapsed = 0;
	//printf(" *** dilate\n");
	elapsed += executeMorphologyKernel(&kernelDilate, src, tmpBuffer);
	//printf(" *** erode\n");
	elapsed += executeMorphologyKernel(&kernelErode, tmpBuffer, dst);

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologyGradient(cl::Buffer& src, cl::Buffer& dst)
{
	//dst = dilate(src) - erode(src);
#if 1
	//printf(" *** gradient\n");
	return executeMorphologyKernel(&kernelGradient, src, dst);

#else
	// Potrzebowac bedziemy dodatkowych buforow tymczasowych
	cl::Buffer tmpBuffer = createBuffer(CL_MEM_READ_WRITE);
	cl::Buffer tmpBuffer2 = createBuffer(CL_MEM_READ_WRITE);

	// dst = src - dilate(erode(src))
	cl_ulong elapsed = 0;
	elapsed += executeMorphologyKernel(&kernelErode, src, tmpBuffer);
	elapsed += executeMorphologyKernel(&kernelDilate, src, tmpBuffer2);
	elapsed += executeSubtractKernel(tmpBuffer2, tmpBuffer, dst);

	return elapsed;
#endif
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologyTopHat(cl::Buffer& src, cl::Buffer& dst)
{
	// Potrzebowac bedziemy dodatkowych buforow tymczasowych
	cl::Buffer tmpBuffer = createBuffer(CL_MEM_READ_WRITE);
	cl::Buffer tmpBuffer2 = createBuffer(CL_MEM_READ_WRITE);

	// dst = src - dilate(erode(src))
	cl_ulong elapsed = 0;
	//printf(" *** erode\n");
	elapsed += executeMorphologyKernel(&kernelErode, src, tmpBuffer);
	//printf(" *** dilate\n");
	elapsed += executeMorphologyKernel(&kernelDilate, tmpBuffer, tmpBuffer2);
	//printf(" *** subtract\n");
	elapsed += executeSubtractKernel(src, tmpBuffer2, dst);

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologyBlackHat(cl::Buffer& src, cl::Buffer& dst)
{
	// Potrzebowac bedziemy dodatkowych buforow tymczasowych
	cl::Buffer tmpBuffer = createBuffer(CL_MEM_READ_WRITE);
	cl::Buffer tmpBuffer2 = createBuffer(CL_MEM_READ_WRITE);

	// dst = close(src) - src
	cl_ulong elapsed = 0;
	//printf(" *** dilate\n");
	elapsed += executeMorphologyKernel(&kernelDilate, src, tmpBuffer);
	//printf(" *** erode\n");
	elapsed += executeMorphologyKernel(&kernelErode, tmpBuffer, tmpBuffer2);
	//printf(" *** subtract\n");
	elapsed += executeSubtractKernel(tmpBuffer2, src, dst);

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologyOutline(cl::Buffer& src, cl::Buffer& dst)
{
	// Skopiuj obraz zrodlowy do docelowego
	cl::Event evt;
	cl_ulong elapsed = copyBuffer(src, dst);

	// Wykonaj operacje hitmiss
	elapsed += executeHitMissKernel(&kernelOutline, src, dst);

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologySkeleton(cl::Buffer& src, 
	cl::Buffer& dst, int& iters)
{
	iters = 0;

	// Potrzebowac bedziemy dodatkowego bufora tymczasowego
	cl::Buffer tmpBuffer = createBuffer(CL_MEM_READ_ONLY);

	// Skopiuj obraz zrodlowy do docelowego i tymczasowego
	cl::Event evt;
	cl_ulong elapsed = 0;
	elapsed += copyBuffer(src, tmpBuffer);
	elapsed += copyBuffer(src, dst);				

	// Licznik atomowy (ew. zwyczajny bufor)
	cl_int err;
	cl_uint d_init = 0;		
	cl::Buffer clAtomicCnt(context, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_uint), &d_init, &err);
	clError("Error while creating temporary OpenCL atomic counter", err);

	do 
	{
		iters++;

		// 8 operacji hit miss, 2 elementy strukturalnego, 4 orientacje
		for(int i = 0; i < 8; ++i)
		{
			elapsed += executeHitMissKernel(&kernelSkeleton_iter[i], 
				tmpBuffer, dst, &clAtomicCnt);

			// Kopiowanie bufora
			elapsed += copyBuffer(dst, tmpBuffer);
		}

		// Sprawdz ile pikseli zostalo zmodyfikowanych
		cl_uint diff;
		elapsed += readAtomicCounter(diff, clAtomicCnt);

		// Sprawdz warunek stopu
		if(diff == 0)
			break;

		printf("Iteration: %3d, pixel changed: %5d\r", iters, diff);

		elapsed += zeroAtomicCounter(clAtomicCnt);

	} while (true);
	printf("\n");

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::morphologySkeletonZhangSuen(cl::Buffer& src,
	cl::Buffer& dst, int& iters)
{
	iters = 0;

	// Skopiuj obraz zrodlowy do docelowego
	cl::Event evt;	
	cl_ulong elapsed = copyBuffer(src, dst);	

	// Licznik atomowy (ew. zwyczajny bufor)
	cl_int err;
	cl_uint d_init = 0;
	cl::Buffer clAtomicCnt(context, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_uint), &d_init, &err);
	clError("Error while creating temporary OpenCL atomic counter", err);

	cl::Buffer clLut(context, 
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(lutTable), lutTable, &err);
	clError("Error while creating temporary OpenCL atomic counter", err);

	// Potrzebowac bedziemy dodatkowego bufora tymczasowego
	cl::Buffer tmpBuffer = createBuffer(CL_MEM_READ_ONLY);

	do 
	{
		iters++;

		// odd pass
		elapsed += copyBuffer(dst, tmpBuffer);
		elapsed += executeHitMissKernel(&kernelSkeleton_pass[0], 
			tmpBuffer, dst, &clLut, &clAtomicCnt);

		// even pass
		elapsed += copyBuffer(dst, tmpBuffer);
		elapsed += executeHitMissKernel(&kernelSkeleton_pass[1], 
			tmpBuffer, dst, &clLut, &clAtomicCnt);

		// Sprawdz ile pikseli zostalo zmodyfikowanych
		cl_uint diff;
		elapsed += readAtomicCounter(diff, clAtomicCnt);

		printf("Iteration: %3d, pixel changed: %5d\r", iters, diff);

		// Sprawdz warunek stopu
		if(diff == 0)
			break;

		elapsed += zeroAtomicCounter(clAtomicCnt);	

	} while(true);
	printf("\n");

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::executeMorphologyKernel(cl::Kernel* kernel, 
	const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer)
{
	cl::Event evt;
	cl_int err;

	cl_int4 seSize = { kradiusx, kradiusy, (int)(csize), 0 };
	cl_int2 imageSize = { sourceBuffer.gpuWidth, sourceBuffer.gpuHeight };

	int apronX = kradiusx * 2;
	int apronY = kradiusy * 2;

	// Ustaw argumenty kernela
	err  = kernel->setArg(0, clSrcBuffer);
	err |= kernel->setArg(1, clDstBuffer);
	err |= kernel->setArg(2, clStructuringElementCoords);
	err |= kernel->setArg(3, seSize);
	err |= kernel->setArg(4, imageSize);
	clError("Error while setting kernel arguments", err);

	int globalItemsX = roundUp(sourceBuffer.cpu->cols - apronX, workGroupSizeX);
	int globalItemsY = roundUp(sourceBuffer.cpu->rows - apronY, workGroupSizeX);

	cl::NDRange offset = cl::NullRange;
	cl::NDRange gridDim(globalItemsX, globalItemsY);
	cl::NDRange blockDim(workGroupSizeX, workGroupSizeY);

	std::string kernelName = kernel->getInfo<CL_KERNEL_FUNCTION_NAME>();
	bool useLocal = kernelName.find("_local") != std::string::npos;

	if(useLocal)
	{
		cl_int2 sharedSize = {
			roundUp(workGroupSizeX + apronX, 4),
			workGroupSizeY + apronY
		};
		size_t sharedBlockSize = sharedSize.s[0] * sharedSize.s[1];
		if(useUint) sharedBlockSize *= sizeof(cl_uint);

		printf("LDS usage (%dx%d): %d B\n", sharedSize.s[0], sharedSize.s[1], sharedBlockSize);

		// Trzeba ustawic dodatkowe argumenty kernela
		err |= kernel->setArg(5, sharedBlockSize, nullptr);
		err |= kernel->setArg(6, sharedSize);
		clError("Error while setting kernel arguments", err);
	}

	// Odpal kernela
	err = cq.enqueueNDRangeKernel(*kernel,
		cl::NullRange, gridDim, blockDim,
		nullptr, &evt);
	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::executeHitMissKernel(cl::Kernel* kernel, 
	const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer, 
	const cl::Buffer* clLut, cl::Buffer* clAtomicCounter)
{
	cl::Event evt;
	cl_int err;

	cl_int2 imageSize = { sourceBuffer.gpuWidth, sourceBuffer.gpuHeight };

	// Ustaw argumenty kernela
	err  = kernel->setArg(0, clSrcBuffer);
	err |= kernel->setArg(1, clDstBuffer);
	err |= kernel->setArg(2, imageSize);
	if(clLut) err |= kernel->setArg(3, *clLut);
	if(clAtomicCounter && clLut) err |= kernel->setArg(4, *clAtomicCounter);
	else if (clAtomicCounter) err |= kernel->setArg(3, *clAtomicCounter);
	clError("Error while setting kernel arguments", err);

	const int lsize = 16;
	int globalItemsX = roundUp(sourceBuffer.cpu->cols - 2, lsize);
	int globalItemsY = roundUp(sourceBuffer.cpu->rows - 2, lsize);

	cl::NDRange offset = cl::NullRange;
	cl::NDRange gridDim(globalItemsX, globalItemsY);
	cl::NDRange blockDim(lsize, lsize);

	// Odpal kernela
	err = cq.enqueueNDRangeKernel(*kernel,
		offset, gridDim, blockDim,
		nullptr, &evt);	

	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::executeSubtractKernel(const cl::Buffer& clABuffer,
	const cl::Buffer& clBBuffer, cl::Buffer& clDstBuffer)
{
	cl_int err;
	cl::Event evt;

	int xitems = sourceBuffer.gpuWidth;
	if(sub4) xitems /= 4;

	cl::NDRange offset = cl::NullRange;
	cl::NDRange blockDim(workGroupSizeX * workGroupSizeY);
	cl::NDRange gridDim(roundUp(xitems * sourceBuffer.gpuHeight, blockDim[0]));

	// Ustaw argumenty kernela
	err  = kernelSubtract.setArg(0, clABuffer);
	err |= kernelSubtract.setArg(1, clBBuffer);
	err |= kernelSubtract.setArg(2, clDstBuffer);
	err |= kernelSubtract.setArg(3, xitems * sourceBuffer.gpuHeight);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	err = cq.enqueueNDRangeKernel(kernelSubtract,
		offset, gridDim, blockDim,
		nullptr, &evt);
	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}