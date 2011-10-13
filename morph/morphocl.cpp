#include "morphocl.h"

#include <QFile>
#include <QTextStream>

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// MorphOpenCL
bool MorphOpenCL::initOpenCL(cl_device_type dt)
{
	// Connect to a compute device
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty())
		return false;

	// FIXME Wybierz platforme 
	cl::Platform platform = platforms[0];
	cl_context_properties properties[] = { 
		CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
		0, 0
	};
	
	// Stworz kontekst
	cl_int err;
	context = cl::Context(dt, properties, nullptr, nullptr, &err);
	clError("Failed to create compute context!", err);

	// Pobierz liste urzadzen
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// FIXME Wybierz pierwsze urzadzenie
	dev = devices[0];
	std::vector<cl::Device> devs(1);
	devs[0] = (dev);

	// Kolejka polecen
	cq = cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	clError("Failed to create command queue!", err);

	return true;
}
// -------------------------------------------------------------------------
void MorphOpenCL::setStructureElement(const cv::Mat& selement)
{
	std::vector<cl_int2> coords;

	kradiusx = (selement.cols - 1) / 2;
	kradiusy = (selement.rows - 1) / 2;

	// Przetworz wstepnie element strukturalny
	for(int y = 0; y < selement.rows; ++y)
	{
		const uchar* krow = selement.data + selement.step*y;

		for(int x = 0; x < selement.cols; ++x)
		{
			if(krow[x] == 0)
				continue;

			cl_int2 c = {x - kradiusx, y - kradiusy};
			coords.push_back(c);
		}
	}
	csize = coords.size();

	// Zaladuj dane do bufora
	cl_int err;
	clSeCoords = cl::Buffer(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		csize * sizeof(cl_int2), coords.data(), &err);
	clError("Error while creating buffer for structure element!", err);
}
// -------------------------------------------------------------------------
void MorphOpenCL::clError(const QString& message, cl_int err)
{
	if(err != CL_SUCCESS && errorCallback != nullptr)
		errorCallback(message, err);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCL::elapsedEvent( const cl::Event& evt )
{
	cl_ulong eventstart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong eventend = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	return (cl_ulong)(eventend - eventstart);
}
// -------------------------------------------------------------------------
cl::Program MorphOpenCL::createProgram(const char* progFile, const char* options)
{
	auto it = programs.find(progFile);

	if(it == programs.end())
	{
		QFile file(progFile);
		if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
		{
			clError("Can't read " + 
				QString(progFile) + 
				" file!", -1);
		}

		QTextStream in(&file);
		QString contents = in.readAll();

		QByteArray w = contents.toLocal8Bit();
		const char* src = w.data();
		size_t len = contents.length();

		cl_int err;
		cl::Program::Sources sources(1, std::make_pair(src, len));
		cl::Program program = cl::Program(context, sources, &err);
		clError("Failed to create compute program from" + QString(progFile), err);

		std::vector<cl::Device> devs(1);
		devs[0] = (dev);

		err = program.build(devs, options);
		if(err != CL_SUCCESS)
		{
			QString log(QString::fromStdString(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev)));
			clError(log, -1);
		}

		programs[progFile] = program;
		return program;
	}
	else
	{
		return it->second;
	}	
}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// MorphOpenCLImage

bool MorphOpenCLImage::initOpenCL(cl_device_type dt)
{
	MorphOpenCL::initOpenCL(dt);

	cl::Program perode = createProgram("kernels-images/erode.cl");
	cl::Program pdilate = createProgram("kernels-images/dilate.cl");
	cl::Program pthinning = createProgram("kernels-images/thinning.cl");
	cl::Program putils = createProgram("kernels-images/utils.cl");

	auto createKernel = [this](const cl::Program& prog,
		const char* kernelName) -> cl::Kernel
	{
		cl_int err;
		cl::Kernel k(prog, kernelName, &err);
		clError("Failed to create " + 
			QString(kernelName) + 
			" kernel!", err);
		return k;
	};

	kernelErode = createKernel(perode, "erode_c4");
	kernelDilate = createKernel(pdilate, "dilate_c4");
	kernelThinning = createKernel(pthinning, "thinning");
	kernelSubtract = createKernel(putils, "subtract");

	return true;
}
// -------------------------------------------------------------------------
void MorphOpenCLImage::setSourceImage(const cv::Mat* src)
{
	cl_int err;
	if(!this->src || this->src->size() != src->size())
	{
		clSrcImage = cl::Image2D(context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
			cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
			src->cols, src->rows, 0, 
			const_cast<uchar*>(src->ptr<uchar>()), &err);
		clError("Error while creating OpenCL source image!", err);
	}
	else
	{
		cl::size_t<3> origin;
		origin[0] = origin[1] = origin[2] = 0;

		cl::size_t<3> region;
		region[0] = src->cols;
		region[1] = src->rows;
		region[2] = 1;

		cl_int err = cq.enqueueWriteImage(clSrcImage, CL_TRUE, 
			origin, region, 0, 0,
			const_cast<uchar*>(src->ptr<uchar>()));
		clError("Error while writing new data to OpenCL source image!", err);
	}

	this->src = src;
}
// -------------------------------------------------------------------------
double MorphOpenCLImage::morphology(EOperationType opType, cv::Mat& dst, int& iters)
{
	// Obraz docelowy
	cl_int err;
	clDstImage = cl::Image2D(context,
		/*CL_MEM_ALLOC_HOST_PTR | */CL_MEM_WRITE_ONLY, 
		cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
		src->cols, src->rows, 0, nullptr, &err); 
	clError("Error while creating destination OpenCL image2D", err);

	iters = 1;
	cl_ulong elapsed = 0;

	// Erozja
	if(opType == OT_Erode)
	{
		elapsed += executeMorphologyKernel(&kernelErode, clSrcImage, clDstImage);
	}
	// Dylatacja
	else if(opType == OT_Dilate)
	{
		elapsed += executeMorphologyKernel(&kernelDilate, clSrcImage, clDstImage);
	}
	else
	{
		auto copyImage = [this](const cl::Image2D& s, cl::Image2D& d,
			cl::Event evt) -> cl_ulong
		{
			cl::size_t<3> origin; origin[0] = origin[1] = origin[2] = 0;
			cl::size_t<3> dorigin; dorigin[0] = dorigin[1] = dorigin[2] = 0;
			cl::size_t<3> region; region[0] = src->cols; region[1] = src->rows; region[2] = 1;

			cq.enqueueCopyImage(s, d, 
				origin, dorigin, region, 
				nullptr, &evt);
			evt.wait();
			return elapsedEvent(evt);
		};

		// Potrzebowac bedziemy dodatkowego bufora tymczasowego
		clTmpImage = cl::Image2D(context,
			CL_MEM_READ_WRITE, 
			cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
			src->cols, src->rows, 0, nullptr, &err); 
		clError("Error while creating temporary OpenCL image2D", err);

		// Otwarcie
		if(opType == OT_Open)
		{
			// dst = dilate(erode(src))
			elapsed += executeMorphologyKernel(&kernelErode, clSrcImage, clTmpImage);
			elapsed += executeMorphologyKernel(&kernelDilate, clTmpImage, clDstImage);
		}
		// Zamkniecie
		else if(opType == OT_Close)
		{
			// dst = erode(dilate(src))
			elapsed += executeMorphologyKernel(&kernelDilate, clSrcImage, clTmpImage);
			elapsed += executeMorphologyKernel(&kernelErode, clTmpImage, clDstImage);
		}
		
		// Operacja wyciagania konturow
		else if(opType == OT_Thinning)
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;
			elapsed += copyImage(clSrcImage, clDstImage, evt);
			elapsed += executeHitMissKernel(&kernelThinning, clSrcImage, clDstImage);
		}
		/*
		// Operacja szkieletyzacji
		else if(opType == OT_Skeleton)
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;	
			elapsed += copyImage(clSrcImage, clTmpImage, evt);
			elapsed += copyImage(clSrcImage, clDstImage, evt);

			// FIXME
			for(int iters = 0; iters < 111; ++iters)
			{
				for(int i = 0; i < 8; ++i)
				{
					elapsed += executeHitMissKernel(&kernelSkeleton_iter[i],
						clTmpImage, clDstImage);

					// Kopiowanie obrazu
					copyImage(clDstImage, clTmpImage, evt);
					elapsed += elapsedEvent(evt);
				}
			}
		}
		*/
		else
		{
			// Potrzebowac bedziemy dodatkowego bufora tymczasowego
			clTmp2Image = cl::Image2D(context,
				CL_MEM_READ_WRITE, 
				cl::ImageFormat(CL_R, CL_UNSIGNED_INT8),
				src->cols, src->rows, 0, nullptr, &err); 
			clError("Error while creating temporary OpenCL image2D", err);

			// Gradient morfologiczny
			if(opType == OT_Gradient)
			{ 
				//dst = dilate(src) - erode(src);
				elapsed += executeMorphologyKernel(&kernelDilate, clSrcImage, clTmpImage);
				elapsed += executeMorphologyKernel(&kernelErode, clSrcImage, clTmp2Image);
				elapsed += executeSubtractKernel(clTmpImage, clTmp2Image, clDstImage);
			}
			// TopHat
			else if(opType == OT_TopHat)
			{ 
				// dst = src - dilate(erode(src))
				elapsed += executeMorphologyKernel(&kernelErode, clSrcImage, clTmpImage);
				elapsed += executeMorphologyKernel(&kernelDilate, clTmpImage, clTmp2Image);
				elapsed += executeSubtractKernel(clSrcImage, clTmp2Image, clDstImage);
			}
			// BlackHat
			else if(opType == OT_BlackHat)
			{ 
				// dst = close(src) - src
				elapsed += executeMorphologyKernel(&kernelDilate, clSrcImage, clTmpImage);
				elapsed += executeMorphologyKernel(&kernelErode, clTmpImage, clTmp2Image);
				elapsed += executeSubtractKernel(clTmp2Image, clSrcImage, clDstImage);
			}
			else
			{
				iters = 0;
				dst.create(src->size(), CV_8U);
				return 0.0;		
			}
		}
	}
	
	// Zczytaj wynik
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;

	cl::size_t<3> region;
	region[0] = src->cols; 
	region[1] = src->rows;
	region[2] = 1;

	dst.create(src->size(), CV_8U);
	cl::Event evt;
	cq.enqueueReadImage(clDstImage, CL_FALSE, origin,
		region, 0, 0, dst.ptr<uchar>(),
		nullptr, &evt);
	evt.wait();

	// Ile czasu zajelo zczytanie danych z powrotem
	elapsed += elapsedEvent(evt);
	// Ile czasu wszystko zajelo
	return elapsed * 0.000001;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::executeMorphologyKernel(cl::Kernel* kernel, 
	const cl::Image2D& clSrcImage, cl::Image2D& clDstImage)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clSrcImage);
	err |= kernel->setArg(1, clDstImage);
	err |= kernel->setArg(2, clSeCoords);
	err |= kernel->setArg(3, csize);
	clError("Error while setting kernel arguments", err);

	//size_t localSize = kernel->getWorkGroupInfo<CL_KERNEL_WORK_GROUP_SIZE>(dev, &err);
	//clError("Error while retrieving kernel work group info!", err);
	//auto w = kernel->getWorkGroupInfo<CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE>(dev);

	// Odpal kernela
	cl::Event evt;	
	cq.enqueueNDRangeKernel(*kernel,
		cl::NullRange,
		cl::NDRange(src->cols, src->rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::executeHitMissKernel(cl::Kernel* kernel, 
	const cl::Image2D& clSrcImage, cl::Image2D& clDstImage)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clSrcImage);
	err |= kernel->setArg(1, clDstImage);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	cl::Event evt;
	cq.enqueueNDRangeKernel(*kernel,
		cl::NDRange(1, 1),
		cl::NDRange(src->cols - 2, src->rows - 2),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

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

	// Odpal kernela
	cl::Event evt;	
	cq.enqueueNDRangeKernel(kernelSubtract,
		cl::NullRange,
		cl::NDRange(src->cols, src->rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// MorphOpenCLBuffer

// -------------------------------------------------------------------------
bool MorphOpenCLBuffer::initOpenCL(cl_device_type dt)
{
	MorphOpenCL::initOpenCL(dt);

	cl::Program perode = createProgram("kernels-buffers/erode.cl");
	cl::Program pdilate = createProgram("kernels-buffers/dilate.cl");
	cl::Program pthinning = createProgram("kernels-buffers/thinning.cl");
	cl::Program putils = createProgram("kernels-buffers/utils.cl");
	cl::Program pskeleton = createProgram("kernels-buffers/skeleton.cl");

	auto createKernel = [this](const cl::Program& prog,
		const char* kernelName) -> cl::Kernel
	{
		cl_int err;
		cl::Kernel k(prog, kernelName, &err);
		clError("Failed to create " + 
			QString(kernelName) + " kernel!", err);
		return k;
	};

	kernelErode = createKernel(perode, "erode");
	kernelDilate = createKernel(pdilate, "dilate");
	kernelThinning = createKernel(pthinning, "thinning");
	kernelSubtract = createKernel(putils, "subtract");

	for(int i = 0; i < 8; ++i)
	{
		QString kernelName = "skeleton_iter" + QString::number(i+1);
		QByteArray kk = kernelName.toAscii();
		const char* k = kk.data();

		kernelSkeleton_iter[i] = createKernel(pskeleton, kk);
	}

	return true;
}
// -------------------------------------------------------------------------
void MorphOpenCLBuffer::setSourceImage(const cv::Mat* src)
{
	cl_int err;
	if(!this->src || this->src->size() != src->size())
	{
		clSrc = cl::Buffer(context,
			CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, 
			src->rows * src->cols, // zakladamy obraz 1-kanalowy
			const_cast<uchar*>(src->ptr<uchar>()), &err);
		clError("Error while creating OpenCL source buffer", err);		
	}
	else
	{
		cl_int err = cq.enqueueWriteBuffer(clSrc, CL_TRUE, 0,
			src->rows * src->cols, src->ptr<uchar>());
		clError("Error while writing new data to OpenCL source buffer!", err);
	}

	this->src = src;
}
// -------------------------------------------------------------------------
double MorphOpenCLBuffer::morphology(EOperationType opType, cv::Mat& dst, int& iters)
{
	size_t dstSize = src->rows * src->cols;

	// Bufor docelowy
	cl_int err;
	clDst = cl::Buffer(context,
		/*CL_MEM_ALLOC_HOST_PTR | */ CL_MEM_WRITE_ONLY, 
		dstSize, // obraz 1-kanalowy
		nullptr, &err);
	clError("Error while creating destination OpenCL buffer", err);

	auto initWithValue = [this](cl::Buffer& buf, int value, size_t size)
	{
		void* ptr = cq.enqueueMapBuffer(buf, CL_TRUE, CL_MAP_WRITE, 0, size);
		memset(ptr, value, size);
		cq.enqueueUnmapMemObject(buf, ptr);
	};
	initWithValue(clDst, 0, dstSize);

	iters = 1;
	cl_ulong elapsed = 0;

	if(opType == OT_Erode)
	{
		elapsed += executeMorphologyKernel(&kernelErode, clSrc, clDst);
	}
	// Dylatacja
	else if(opType == OT_Dilate)
	{
		elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clDst);
	}
	else
	{
		// Funkcja lambda kopiujaca zawartosc jednego bufora OpenCL'a do drugiego
		// przy okazji mierzac czas tej operacji
		auto copyBuffer = [this](const cl::Buffer& clsrc, cl::Buffer& cldst,
			cl::Event& clevt) -> cl_ulong
		{
			cq.enqueueCopyBuffer(clsrc, cldst, 
				0, 0, src->size().area(), 
				nullptr, &clevt);
			clevt.wait();
			return elapsedEvent(clevt);
		};

		// Potrzebowac bedziemy dodatkowego bufora tymczasowego
		clTmp = cl::Buffer(context,
			CL_MEM_READ_WRITE, 
			dstSize, // obraz 1-kanalowy
			nullptr, &err);
		clError("Error while creating temporary OpenCL buffer", err);
		initWithValue(clTmp, 0, dstSize);

		// Otwarcie
		if(opType == OT_Open)
		{
			// dst = dilate(erode(src))
			elapsed += executeMorphologyKernel(&kernelErode, clSrc, clTmp);
			elapsed += executeMorphologyKernel(&kernelDilate, clTmp, clDst);
		}
		// Zamkniecie
		else if(opType == OT_Close)
		{
			// dst = erode(dilate(src))
			elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clTmp);
			elapsed += executeMorphologyKernel(&kernelErode, clTmp, clDst);
		}
		// Operacja wyciagania konturow
		else if(opType == OT_Thinning)
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;
			elapsed += copyBuffer(clSrc, clDst, evt);
			elapsed += executeHitMissKernel(&kernelThinning, clSrc, clDst);
		}
		// Operacja szkieletyzacji
		else if(opType == OT_Skeleton)
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;	
			elapsed += copyBuffer(clSrc, clTmp, evt);
			elapsed += copyBuffer(clSrc, clDst, evt);

			for(int iters = 0; iters < 111; ++iters)
			{
				for(int i = 0; i < 8; ++i)
				{
					elapsed += executeHitMissKernel(&kernelSkeleton_iter[i], clTmp, clDst);

					// Kopiowanie bufora
					copyBuffer(clDst, clTmp, evt);
					elapsed += elapsedEvent(evt);
				}
			}
		}
		else
		{
			// Potrzebowac bedziemy dodatkowego bufora tymczasowego
			clTmp2 = cl::Buffer(context,
				CL_MEM_READ_WRITE, 
				dstSize, // obraz 1-kanalowy
				nullptr, &err);
			clError("Error while creating temporary OpenCL buffer", err);
			initWithValue(clTmp2, 0, dstSize);

			// Gradient morfologiczny
			if(opType == OT_Gradient)
			{ 
				//dst = dilate(src) - erode(src);
				elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clTmp);
				elapsed += executeMorphologyKernel(&kernelErode, clSrc, clTmp2);
				elapsed += executeSubtractKernel(clTmp, clTmp2, clDst);
			}
			// TopHat
			else if(opType == OT_TopHat)
			{ 
				// dst = src - dilate(erode(src))
				elapsed += executeMorphologyKernel(&kernelErode, clSrc, clTmp);
				elapsed += executeMorphologyKernel(&kernelDilate, clTmp, clTmp2);
				elapsed += executeSubtractKernel(clSrc, clTmp2, clDst);
			}
			// BlackHat
			else if(opType == OT_BlackHat)
			{ 
				// dst = close(src) - src
				elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clTmp);
				elapsed += executeMorphologyKernel(&kernelErode, clTmp, clTmp2);
				elapsed += executeSubtractKernel(clTmp2, clSrc, clDst);
			}
			else
			{
				iters = 0;
				dst.create(src->size(), CV_8U);
				return 0.0;				
			}
		}
	}

	// Zczytaj wynik z karty
	dst.create(src->size(), CV_8U);
	cl::Event evt;
	cq.enqueueReadBuffer(clDst, CL_FALSE, 0,
		dstSize, dst.ptr<uchar>(),
		nullptr, &evt);
	evt.wait();

	// Ile czasu zajelo zczytanie danych z powrotem
	elapsed += elapsedEvent(evt);
	// Ile czasu wszystko zajelo
	return elapsed * 0.000001;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::executeMorphologyKernel(cl::Kernel* kernel, 
	const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clSrcBuffer);
	err |= kernel->setArg(1, clDstBuffer);
	err |= kernel->setArg(2, clSeCoords);
	err |= kernel->setArg(3, csize);
	err |= kernel->setArg(4, src->cols);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	cl::Event evt;	
	cq.enqueueNDRangeKernel(*kernel,
		cl::NDRange(kradiusx, kradiusy),
		cl::NDRange(src->cols - kradiusx*2, src->rows - kradiusy*2),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::executeHitMissKernel(cl::Kernel* kernel, 
	const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clSrcBuffer);
	err |= kernel->setArg(1, clDstBuffer);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	cl::Event evt;
	cq.enqueueNDRangeKernel(*kernel,
		cl::NDRange(1, 1),
		cl::NDRange(src->cols - 2, src->rows - 2),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::executeSubtractKernel(const cl::Buffer& clABuffer,
	const cl::Buffer& clBBuffer, cl::Buffer& clDstBuffer)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernelSubtract.setArg(0, clABuffer);
	err |= kernelSubtract.setArg(1, clBBuffer);
	err |= kernelSubtract.setArg(2, clDstBuffer);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	cl::Event evt;	
	cq.enqueueNDRangeKernel(kernelSubtract,
		cl::NullRange,
		cl::NDRange(src->cols * src->rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
