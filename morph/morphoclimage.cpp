#include "morphoclimage.h"

#include <QSettings>

MorphOpenCLImage::MorphOpenCLImage()
	: MorphOpenCL()
{
}
// -------------------------------------------------------------------------
MorphOpenCLImage::~MorphOpenCLImage()
{
}
// -------------------------------------------------------------------------
bool MorphOpenCLImage::initOpenCL()
{
	MorphOpenCL::initOpenCL();
	if(error) return false;

	// Pobierz obslugiwane formaty obrazow
	std::vector<cl::ImageFormat> imageFormats;
	context.getSupportedImageFormats(CL_MEM_READ_WRITE,
		CL_MEM_OBJECT_IMAGE2D, &imageFormats);

	// Sprawdz czy ten co chcemy jest obslugiwany
	bool found = false;
	sourceImage.format.image_channel_data_type = CL_UNORM_INT8;
	sourceImage.format.image_channel_order = CL_R;

	for(auto i = imageFormats.cbegin(), 
		ie = imageFormats.cend(); i != ie; ++i)
	{
		if (i->image_channel_data_type == sourceImage.format.image_channel_data_type &&
			i->image_channel_order == sourceImage.format.image_channel_order)
		{
			found = true;
			break;
		}
	}

	if(!found)
	{
		clError("Required image format (CL_R, CL_UNSIGNED8) not supported!",
			CL_IMAGE_FORMAT_NOT_SUPPORTED);
	}
	if(error) return false;

	QSettings s("./settings.cfg", QSettings::IniFormat);
	QString opts = "-Ikernels-buffer2D/";
	
	if(s.value("opencl/atomiccounters", false).toBool())
	{
		opts += " -DUSE_ATOMIC_COUNTERS";
		printf("Using atomic counters instead of global atomic operations\n");
	}

	// do ewentualnej rekompilacji z podaniem innego parametry -D
	erodeParams.programName = "kernels-buffer2D/erode.cl";
	erodeParams.options = opts;
	erodeParams.kernelName = s.value("kernel-buffer2D/erode", "erode").toString();
	erodeParams.needRecompile = erodeParams.kernelName.contains("_pragma", Qt::CaseSensitive);

	dilateParams.programName = "kernels-buffer2D/dilate.cl";
	dilateParams.options = opts;
	dilateParams.kernelName = s.value("kernel-buffer2D/dilate", "dilate").toString();
	dilateParams.needRecompile = dilateParams.kernelName.contains("_pragma", Qt::CaseSensitive);

	gradientParams.programName = "kernels-buffer2D/gradient.cl";
	gradientParams.options = opts;
	gradientParams.kernelName = s.value("kernel-buffer2D/gradient", "gradient").toString();
	gradientParams.needRecompile = gradientParams.kernelName.contains("_pragma", Qt::CaseSensitive);

	// Wczytaj programy (rekompilowalne)
	cl::Program perode = createProgram(erodeParams.programName, opts);
	cl::Program pdilate = createProgram(dilateParams.programName, opts);
	cl::Program pgradient = createProgram(gradientParams.programName, opts);

	// Wczytaj reszte programow (nie ma sensu ich rekompilowac)
	cl::Program poutline = createProgram("kernels-buffer2D/outline.cl", opts);
	cl::Program putils = createProgram("kernels-buffer2D/utils.cl", opts);
	cl::Program pskeleton = createProgram("kernels-buffer2D/skeleton.cl", opts);
	cl::Program pskeletonz = createProgram("kernels-buffer2D/skeleton_zhang.cl", opts);

	// Stworz kernele (nazwy pobierz z pliku konfiguracyjnego)
	kernelErode = createKernel(perode, erodeParams.kernelName);
	kernelDilate = createKernel(pdilate, dilateParams.kernelName);
	kernelGradient = createKernel(pgradient, gradientParams.kernelName);
	
	kernelOutline = createKernel(poutline, "outline");
	kernelSubtract = createKernel(putils, "subtract");

	for(int i = 0; i < 8; ++i)
	{
		QString kernelName = "skeleton_iter" + QString::number(i+1);
		kernelSkeleton_iter[i] = createKernel(pskeleton, kernelName);
	}

	kernelSkeleton_pass[0]  = createKernel(pskeletonz, "skeletonZhang_pass1");
	kernelSkeleton_pass[1]  = createKernel(pskeletonz, "skeletonZhang_pass2");

	return true;
}
// -------------------------------------------------------------------------
void MorphOpenCLImage::setSourceImage(const cv::Mat* newSrc)
{
	cl_int err;
	sourceImage.cpu = newSrc;

	// Zaladuj obraz zrodlowy do karty
	sourceImage.gpu = cl::Image2D(context,
		CL_MEM_READ_ONLY, sourceImage.format,
		sourceImage.cpu->cols, 
		sourceImage.cpu->rows, 0, 
		nullptr, &err);
	clError("Error while creating OpenCL source image!", err);

	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;

	cl::size_t<3> region;
	region[0] = sourceImage.cpu->cols;
	region[1] = sourceImage.cpu->rows;
	region[2] = 1;

	cl::Event evt;
	err = cq.enqueueWriteImage(sourceImage.gpu, CL_FALSE, 
		origin, region, 0, 0,
		const_cast<uchar*>(sourceImage.cpu->ptr<uchar>()),
		0, &evt);
	clError("Error while writing new data to OpenCL source image!", err);
	evt.wait();

	// Podaj czas trwania transferu
	cl_ulong delta = elapsedEvent(evt);
	printf("Transfering source image to GPU took %.05lf ms\n", delta * 0.000001);	
}
// -------------------------------------------------------------------------
void MorphOpenCLImage::setSourceImage(const cv::Mat* newSrc, GLuint glresource)
{
	setSourceImage(newSrc);

	if(newSrc->cols != sharedw || newSrc->rows != sharedh)
	{
		sharedw = newSrc->cols;
		sharedh = newSrc->rows;

		cl_int err;
		shared = cl::Image2DGL(context, CL_MEM_WRITE_ONLY, GL_TEXTURE_2D, 0, glresource, &err);
		clError("Can't create shared GL/CL 2D buffer", err);
	}
}
// -------------------------------------------------------------------------
double MorphOpenCLImage::morphology(EOperationType opType, cv::Mat& dst, int& iters)
{
	iters = 1;
	cl_ulong elapsed = 0;

	// Obraz docelowy
	cl::Image2D clDstImage;
	if(useShared) clDstImage = shared;
	else clDstImage = createImage2D(CL_MEM_WRITE_ONLY);

	switch(opType)
	{
	case OT_Erode:
		elapsed += morphologyErode(sourceImage.gpu, clDstImage);
		break;
	case OT_Dilate:
		elapsed += morphologyDilate(sourceImage.gpu, clDstImage);
		break;
	case OT_Open:
		elapsed += morphologyOpen(sourceImage.gpu, clDstImage);
		break;
	case OT_Close:
		elapsed += morphologyClose(sourceImage.gpu, clDstImage);
		break;
	case OT_Gradient:
		elapsed += morphologyGradient(sourceImage.gpu, clDstImage);
		break;
	case OT_TopHat:
		elapsed += morphologyTopHat(sourceImage.gpu, clDstImage);
		break;
	case OT_BlackHat:
		elapsed += morphologyBlackHat(sourceImage.gpu, clDstImage);
		break;
	case OT_Outline:
		elapsed += morphologyOutline(sourceImage.gpu, clDstImage);
		break;
	case OT_Skeleton:
		elapsed += morphologySkeleton(sourceImage.gpu, clDstImage, iters);
		break;
	case OT_Skeleton_ZhangSuen:
		elapsed += morphologySkeletonZhangSuen(sourceImage.gpu, clDstImage, iters);
		break;
	}

	// Zczytaj wynik z karty (tylko w przypadku nie dzielenia zasobu)
	if(!useShared)
	{
		cl_ulong readingTime = readBack(clDstImage, dst);

		double totalTime = (elapsed + readingTime) * 0.000001;
		printf("Total time: %.05lf ms (in which %.05f was a processing time "
			"and %.05lf ms was a transfer time)\n",
			totalTime, elapsed * 0.000001, readingTime * 0.000001);

		// Ile czasu wszystko zajelo
		return totalTime;
	}
	else
	{
		double totalTime = elapsed * 0.000001;
		printf("Total time: %.05lf ms (+ 0 transfer time)\n", totalTime);

		// Ile czasu wszystko zajelo
		return totalTime;
	}
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::readBack(cl::Image2D& source, cv::Mat &dst)
{
	dst = cv::Mat(sourceImage.cpu->size(), CV_8U, cv::Scalar(0));

	// Zczytaj wynik z karty
	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	// Chcemy caly obszar
	cl::size_t<3> region;
	region[0] = sourceImage.cpu->cols;
	region[1] = sourceImage.cpu->rows;
	region[2] = 1;

	cl::Event evt;
	cl_int err = cq.enqueueReadImage(source, CL_FALSE, 
		origin, region, 0, 0, 
		dst.ptr<uchar>(), nullptr, &evt);
	clError("Error while reading result to image buffer!", err);
	evt.wait();

	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::copyImage2D(const cl::Image2D& src, cl::Image2D& dst)
{
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;

	cl::size_t<3> region; 
	region[0] = sourceImage.cpu->cols; 
	region[1] = sourceImage.cpu->rows;
	region[2] = 1;

	cl::Event evt;
	cq.enqueueCopyImage(src, dst, 
		origin, origin, region, 
		nullptr, &evt);
	evt.wait();
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl::Image2D MorphOpenCLImage::createImage2D(cl_mem_flags memFlags)
{
	cl_int err;
	cl::Image2D img = cl::Image2D(context, 
		memFlags, sourceImage.format, 
		sourceImage.cpu->cols, 
		sourceImage.cpu->rows, 0, nullptr, &err);
	clError("Error while creating OpenCL image2D.", err);
	return img;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::morphologyErode(cl::Image2D& src, cl::Image2D& dst)
{
	return executeMorphologyKernel(&kernelErode, src, dst);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::morphologyDilate(cl::Image2D& src, cl::Image2D& dst)
{
	return executeMorphologyKernel(&kernelDilate, src, dst);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::morphologyOpen(cl::Image2D& src, cl::Image2D& dst)
{
	// Potrzebowac bedziemy dodatkowego bufora tymczasowego
	cl::Image2D tmpImage = createImage2D(CL_MEM_READ_WRITE);

	// dst = dilate(erode(src))
	cl_ulong elapsed = 0;
	elapsed += executeMorphologyKernel(&kernelErode, src, tmpImage);
	elapsed += executeMorphologyKernel(&kernelDilate, tmpImage, dst);

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::morphologyClose(cl::Image2D& src, cl::Image2D& dst)
{
	// Potrzebowac bedziemy dodatkowego bufora tymczasowego
	cl::Image2D tmpImage = createImage2D(CL_MEM_READ_WRITE);

	// dst = erode(dilate(src))
	cl_ulong elapsed = 0;
	elapsed += executeMorphologyKernel(&kernelDilate, src, tmpImage);
	elapsed += executeMorphologyKernel(&kernelErode, tmpImage, dst);

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::morphologyGradient(cl::Image2D& src, cl::Image2D& dst)
{
#if 1
	return executeMorphologyKernel(&kernelGradient, src, dst);
#else
	// Potrzebowac bedziemy dodatkowych buforow tymczasowych
	cl::Image2D tmpImage = createImage2D(CL_MEM_READ_WRITE);
	cl::Image2D tmpImage2 = createImage2D(CL_MEM_READ_WRITE);

	// dst = src - dilate(erode(src))
	cl_ulong elapsed = 0;
	elapsed += executeMorphologyKernel(&kernelErode, src, tmpImage);
	elapsed += executeMorphologyKernel(&kernelDilate, src, tmpImage2);
	elapsed += executeSubtractKernel(tmpImage2, tmpImage, dst);
	return elapsed;
#endif
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::morphologyTopHat(cl::Image2D& src, cl::Image2D& dst)
{
	// Potrzebowac bedziemy dodatkowych buforow tymczasowych
	cl::Image2D tmpImage = createImage2D(CL_MEM_READ_WRITE);
	cl::Image2D tmpImage2 = createImage2D(CL_MEM_READ_WRITE);

	// dst = src - dilate(erode(src))
	cl_ulong elapsed = 0;
	elapsed += executeMorphologyKernel(&kernelErode, src, tmpImage);
	elapsed += executeMorphologyKernel(&kernelDilate, tmpImage, tmpImage2);
	elapsed += executeSubtractKernel(src, tmpImage2, dst);
	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::morphologyBlackHat(cl::Image2D& src, cl::Image2D& dst)
{
	// Potrzebowac bedziemy dodatkowych buforow tymczasowych
	cl::Image2D tmpImage = createImage2D(CL_MEM_READ_WRITE);
	cl::Image2D tmpImage2 = createImage2D(CL_MEM_READ_WRITE);

	// dst = close(src) - src
	cl_ulong elapsed = 0;
	elapsed += executeMorphologyKernel(&kernelDilate, src, tmpImage);
	elapsed += executeMorphologyKernel(&kernelErode, tmpImage, tmpImage2);
	elapsed += executeSubtractKernel(tmpImage2, src, dst);
	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::morphologyOutline(cl::Image2D& src, cl::Image2D& dst)
{
	// Skopiuj obraz zrodlowy do docelowego
	cl::Event evt;
	cl_ulong elapsed = copyImage2D(src, dst);

	// Wykonaj operacje hitmiss
	elapsed += executeHitMissKernel(&kernelOutline, src, dst);

	return elapsed;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::morphologySkeleton(cl::Image2D& src, cl::Image2D& dst,
	int& iters)
{
	iters = 0;

	// Potrzebowac bedziemy dodatkowego bufora tymczasowego
	cl::Image2D tmpImage = createImage2D(CL_MEM_READ_ONLY);

	// Skopiuj obraz zrodlowy do docelowego i tymczasowego
	cl::Event evt;
	cl_ulong elapsed = 0;
	elapsed += copyImage2D(src, tmpImage);
	elapsed += copyImage2D(src, dst);

	// Licznik atomowy (ew. zwyczajny bufor)
	cl_int err;
	cl_uint d_init = 0;
	cl::Buffer clAtomicCnt(context, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_uint), &d_init, &err);
	clError("Error while creating temporary OpenCL atomic counter", err);
	if(err != CL_SUCCESS)
		return 0;

	do 
	{
		iters++;

		// 8 operacji hit miss, 2 elementy strukturalnego, 4 orientacje
		for(int i = 0; i < 8; ++i)
		{
			elapsed += executeHitMissKernel(&kernelSkeleton_iter[i],
				tmpImage, dst, &clAtomicCnt);

			// Kopiowanie obrazu
			elapsed += copyImage2D(dst, tmpImage);
		}

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
cl_ulong MorphOpenCLImage::morphologySkeletonZhangSuen(cl::Image2D& src, 
	cl::Image2D& dst, int& iters)
{
	iters = 0;

	// Skopiuj obraz zrodlowy do docelowego
	cl::Event evt;
	cl_ulong elapsed = copyImage2D(src, dst);

	// Licznik atomowy (ew. zwyczajny bufor)
	cl_int err;
	cl_uint d_init = 0;
	cl::Buffer clAtomicCnt(context, 
		CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, 
		sizeof(cl_uint), &d_init, &err);
	clError("Error while creating temporary OpenCL atomic counter", err);
	if(err != CL_SUCCESS)
		return 0;

	cl::Buffer clLut(context, 
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		sizeof(lutTable), lutTable, &err);
	clError("Error while creating temporary OpenCL atomic counter", err);
	if(err != CL_SUCCESS)
		return 0;

	// Potrzebowac bedziemy dodatkowego bufora tymczasowego
	cl::Image2D tmpImage = createImage2D(CL_MEM_READ_ONLY);

	do 
	{
		iters++;

		// odd pass
		elapsed += copyImage2D(dst, tmpImage);
		elapsed += executeHitMissKernel(&kernelSkeleton_pass[0], 
			tmpImage, dst, &clLut, &clAtomicCnt);

		// even pass
		elapsed += copyImage2D(dst, tmpImage);
		elapsed += executeHitMissKernel(&kernelSkeleton_pass[1], 
			tmpImage, dst, &clLut, &clAtomicCnt);

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
cl_ulong MorphOpenCLImage::executeMorphologyKernel(cl::Kernel* kernel, 
	const cl::Image2D& clSrcImage, cl::Image2D& clDstImage)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clSrcImage);
	err |= kernel->setArg(1, clDstImage);
	err |= kernel->setArg(2, clStructuringElementCoords);
	err |= kernel->setArg(3, csize);
	clError("Error while setting kernel arguments", err);

	if(err != CL_SUCCESS)
		return 0;

	cl::NDRange offset = cl::NullRange;
	cl::NDRange gridDim(
		roundUp(sourceImage.cpu->cols, workGroupSizeX),
		roundUp(sourceImage.cpu->rows, workGroupSizeY));
	cl::NDRange blockDim(workGroupSizeX, workGroupSizeY);

	// Odpal kernela
	cl::Event evt;	
	err |= cq.enqueueNDRangeKernel(*kernel,
		offset, gridDim, blockDim,
		nullptr, &evt);
	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::executeHitMissKernel(cl::Kernel* kernel, 
	const cl::Image2D& clSrcImage, cl::Image2D& clDstImage,
	const cl::Buffer* clLut, cl::Buffer* clAtomicCnt)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clSrcImage);
	err |= kernel->setArg(1, clDstImage);

	if(clLut) err |= kernel->setArg(2, *clLut);
	if(clAtomicCnt && clLut) err |= kernel->setArg(3, *clAtomicCnt);
	else if (clAtomicCnt) err |= kernel->setArg(2, *clAtomicCnt);

	clError("Error while setting kernel arguments", err);
	if(err != CL_SUCCESS)
		return 0;

	cl::NDRange offset(1, 1);
	cl::NDRange gridDim(
		roundUp(sourceImage.cpu->cols - 2, workGroupSizeX),
		roundUp(sourceImage.cpu->rows - 2, workGroupSizeY));
	cl::NDRange blockDim(workGroupSizeX, workGroupSizeY);

	// Odpal kernela
	cl::Event evt;
	err |= cq.enqueueNDRangeKernel(*kernel,
		offset, gridDim, blockDim, 
		nullptr, &evt);
	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::executeSubtractKernel(const cl::Image2D& clAImage,
	const cl::Image2D& clBImage, cl::Image2D& clDstImage)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernelSubtract.setArg(0, clAImage);
	err |= kernelSubtract.setArg(1, clBImage);
	err |= kernelSubtract.setArg(2, clDstImage);
	clError("Error while setting kernel arguments", err);
	if(err != CL_SUCCESS)
		return 0;

	cl::NDRange offset = cl::NullRange;
	cl::NDRange gridDim(
		roundUp(sourceImage.cpu->cols, workGroupSizeX),
		roundUp(sourceImage.cpu->rows, workGroupSizeY));
	cl::NDRange blockDim(workGroupSizeX, workGroupSizeY);

	// Odpal kernela
	cl::Event evt;	
	err |= cq.enqueueNDRangeKernel(kernelSubtract,
		offset, gridDim, blockDim, 
		nullptr, &evt);
	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
