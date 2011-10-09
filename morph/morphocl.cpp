#include "morphocl.h"

bool MorphOpenCL::initOpenCL()
{
	// Connect to a compute device
	std::vector<cl::Platform> platforms;
	cl::Platform::get(&platforms);

	if (platforms.empty())
	{
		QMessageBox::critical(nullptr,
			"Critical error",
			"No OpenCL Platform available therefore OpenCL processing will be disabled",
			QMessageBox::Ok);
		return false;
	}

	// FIXME
	cl::Platform platform = platforms[0];
	cl_context_properties properties[] = { 
		CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
		0, 0
	};

	// Tylko GPU (i tak CPU chwilowo AMD uwalil)
	cl_int err;
	context = cl::Context(CL_DEVICE_TYPE_CPU, properties, nullptr, nullptr, &err);
	clError("Failed to create compute context!", err);

	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// FIXME
	dev = devices[0];
	std::vector<cl::Device> devs(1);
	devs[0] = (dev);

	//
	//cl_bool imageSupport = dev.getInfo<CL_DEVICE_IMAGE_SUPPORT>();
	//

	// Kolejka polecen
	cq = cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	clError("Failed to create command queue!", err);

	// Zaladuj Kernele
	QFile file("./kernels-uchar.cl");
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
		clError("Can't read kernels-uchar.cl file", -1);

	QTextStream in(&file);
	QString contents = in.readAll();

	QByteArray w = contents.toLocal8Bit();
	const char* src = w.data();
	size_t len = contents.length();

	cl::Program::Sources sources(1, std::make_pair(src, len));
	cl::Program program = cl::Program(context, sources, &err);
	clError("Failed to create compute program!", err);

	err = program.build(devs);
	if(err != CL_SUCCESS)
	{
		QString log(QString::fromStdString(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev)));
		clError(log, -1);
	}

	// Stworz kernele ze zbudowanego programu
	kernelSubtract = cl::Kernel(program, "subtract", &err);
	clError("Failed to create dilate kernel!", err);

	kernelAddHalf = cl::Kernel(program, "addHalf", &err);
	clError("Failed to create addHalf kernel!", err);

	kernelErode = cl::Kernel(program, "erode", &err);
	clError("Failed to create erode kernel!", err);

	kernelDilate = cl::Kernel(program, "dilate", &err);
	clError("Failed to create dilate kernel!", err);

	kernelRemove = cl::Kernel(program, "remove", &err);
	clError("Failed to create remove kernel!", err);

	for(int i = 0; i < 8; ++i)
	{
		QString kernelName = "skeleton_iter" + QString::number(i+1);
		QByteArray kk = kernelName.toAscii();
		const char* k = kk.data();
		kernelSkeleton_iter[i] = cl::Kernel(program, k, &err);
		clError("Failed to create skeleton_iter kernel!", err);
	}

	return true;
}
// -------------------------------------------------------------------------
void MorphOpenCL::setSourceImage(const cv::Mat* src)
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
		clError("Error while writing new data to OpenCL buffer!", err);
	}

	this->src = src;
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
double MorphOpenCL::morphology(EOperationType opType, cv::Mat& dst, int& iters)
{
	size_t dstSize = src->rows * src->cols;

	// Bufor docelowy
	cl_int err;
	clDst = cl::Buffer(context,
		CL_MEM_ALLOC_HOST_PTR | CL_MEM_WRITE_ONLY, 
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
		else if(opType == OT_Remove)
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;
			elapsed += copyBuffer(clSrc, clDst, evt);
			elapsed += executeHitMissKernel(&kernelRemove, clSrc, clDst);
		}
		// Operacja szkieletyzacji
		else if(opType == OT_Skeleton)
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;	
			copyBuffer(clSrc, clTmp, evt);
			copyBuffer(clSrc, clDst, evt);

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

			elapsed += executeAddHalfKernel(clSrc, clDst);
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
	double delapsed = elapsed * 0.000001;

	return delapsed;
}
// -------------------------------------------------------------------------
void MorphOpenCL::clError(const QString& message, cl_int err)
{
	if(err != CL_SUCCESS && errorCallback != nullptr)
		errorCallback(message, err);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCL::executeMorphologyKernel(cl::Kernel* kernel, 
	const cl::Buffer& clBufferSrc, cl::Buffer& clBufferDst)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clBufferSrc);
	err |= kernel->setArg(1, clBufferDst);
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
cl_ulong MorphOpenCL::executeHitMissKernel(cl::Kernel* kernel, 
	const cl::Buffer& clBufferSrc, cl::Buffer& clBufferDst)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clBufferSrc);
	err |= kernel->setArg(1, clBufferDst);
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
cl_ulong MorphOpenCL::executeSubtractKernel(const cl::Buffer& clBufferA,
	const cl::Buffer& clBufferB, cl::Buffer& clBufferDst)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernelSubtract.setArg(0, clBufferA);
	err |= kernelSubtract.setArg(1, clBufferB);
	err |= kernelSubtract.setArg(2, clBufferDst);
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
// -------------------------------------------------------------------------
cl_ulong MorphOpenCL::executeAddHalfKernel(const cl::Buffer& clBufferSrc,
	cl::Buffer& clBufferDst)
{
	cl_int err;
	err  = kernelAddHalf.setArg(0, clBufferDst);
	err |= kernelAddHalf.setArg(1, clBufferSrc);
	clError("Error while setting kernel arguments", err);

	// Odpal kernela
	cl::Event evt;
	cq.enqueueNDRangeKernel(kernelAddHalf,
		cl::NullRange,
		cl::NDRange(src->cols * src->rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCL::elapsedEvent( const cl::Event& evt )
{
	cl_ulong eventstart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong eventend = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	return (cl_ulong)(eventend - eventstart);
}