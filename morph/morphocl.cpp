#include "morphocl.h"

#include <QFile>
#include <QTextStream>
#include <QSettings>

int roundUp(int value, int multiple)
{
	int v = value % multiple;
	return value + (multiple - v);
}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// MorphOpenCL

MorphOpenCL::MorphOpenCL()
: errorCallback(nullptr),
src(nullptr),
kradiusx(0),
kradiusy(0)
{
	QSettings settings("./settings.cfg", QSettings::IniFormat);

	// Wczytaj opcje z pliku konfiguracyjnego
	workGroupSizeX = settings.value("misc/workgroupsizex", 16).toInt();
	workGroupSizeY = settings.value("misc/workgroupsizey", 16).toInt();
	readingMethod = static_cast<EReadingMethod>(
		settings.value("misc/readingmethod", 0).toInt());
	local = settings.value("kernel/local", false).toBool();
}
// -------------------------------------------------------------------------
bool MorphOpenCL::initOpenCL()
{
	// Connect to a compute device
	std::vector<cl::Platform> platforms;
	cl_int err;
	err = cl::Platform::get(&platforms);

	if (platforms.empty())
		return false;

	cl::Platform platform;

	// Jesli mamy tylko jedna platforme to ja wybierz bez pytania
	if(platforms.size() == 1)
	{
		platform = platforms[0];
		std::string name = platform.getInfo<CL_PLATFORM_NAME>(&err);
		std::string version = platform.getInfo<CL_PLATFORM_VERSION>(&err);
		printf("Using '%s' %s platform.\n", name.c_str(), version.c_str());
	}
	else
	{
		for(size_t i = 0; i < platforms.size(); ++i)
		{
			std::string name = platforms[i].getInfo<CL_PLATFORM_NAME>(&err);
			std::string version = platforms[i].getInfo<CL_PLATFORM_VERSION>(&err);
			printf("\t%d) %s %s\n", i+1, name.c_str(), version.c_str());
		}

		int choice = 0;
		while(choice > platforms.size() || choice <= 0)
		{
			printf("\nChoose OpenCL platform: \n");
			scanf("%d", &choice);
		}
		platform = platforms[choice-1];
	}

	cl_context_properties properties[] = { 
		CL_CONTEXT_PLATFORM, (cl_context_properties)(platform)(),
		0, 0
	};	
	
	// Stworz kontekst
	context = cl::Context(CL_DEVICE_TYPE_CPU, properties, nullptr, nullptr, &err);
	clError("Failed to create compute context!", err);

	// Pobierz liste urzadzen
	std::vector<cl::Device> devices = context.getInfo<CL_CONTEXT_DEVICES>();

	// Wybierz urzadzenie
	if(devices.size() == 1)
	{
		dev = devices[0];
		std::string devName = dev.getInfo<CL_DEVICE_NAME>(&err);
		printf("Using %s\n", devName.c_str());
	}
	else
	{
		printf("\n");
		for(size_t i = 0; i < devices.size(); ++i)
		{
			std::string devName = devices[i].getInfo<CL_DEVICE_NAME>(&err);
			cl_bool devImagesSupported = devices[i].getInfo<CL_DEVICE_IMAGE_SUPPORT>(&err);
			printf("\t%d) %s (%s)\n", i+1, devName.c_str(), 
				(devImagesSupported ? "Images supported" : "Images NOT supported"));
		}

		int choice = 0;
		while(choice > devices.size() || choice <= 0)
		{
			printf("\nChoose OpenCL device: \n");
			scanf("%d", &choice);
		}
		dev = devices[choice-1];
	}

	std::vector<cl::Device> devs(1);
	devs[0] = (dev);

	// Kolejka polecen
	cq = cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	clError("Failed to create command queue!", err);

	return true;
}
// -------------------------------------------------------------------------
int MorphOpenCL::setStructureElement(const cv::Mat& selement)
{
	std::vector<cl_int2> coords;

	kradiusx = (selement.cols - 1) / 2;
	kradiusy = (selement.rows - 1) / 2;

	// Dla implementacji z wykorzystaniem obrazow musimy przesunac uklad wspolrzednych
	// elementu strukturalnego
	bool shiftCoords = (dynamic_cast<MorphOpenCLImage*>(this)) != nullptr;

	// Przetworz wstepnie element strukturalny
	for(int y = 0; y < selement.rows; ++y)
	{
		const uchar* krow = selement.data + selement.step*y;

		for(int x = 0; x < selement.cols; ++x)
		{
			if(krow[x] == 0)
				continue;

			cl_int2 c = {x, y};
			if(shiftCoords)
			{ 
				c.s[0] -= kradiusx;
				c.s[1] -= kradiusy;
			}
			
			coords.push_back(c);
		}
	}
	csize = static_cast<int>(coords.size());
	printf("Structure element size (number of 'white' pixels): %d\n", csize);

	// Zaladuj dane do bufora
	cl_int err;
	clSeCoords = cl::Buffer(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		csize * sizeof(cl_int2), coords.data(), &err);
	clError("Error while creating buffer for structure element!", err);

	return csize;
}
// -------------------------------------------------------------------------
void MorphOpenCL::clError(const QString& message, cl_int err)
{
	if(err != CL_SUCCESS && errorCallback != nullptr)
		errorCallback(message, err);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCL::elapsedEvent(const cl::Event& evt)
{
	cl_ulong eventstart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong eventend = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	return (cl_ulong)(eventend - eventstart);
}
// -------------------------------------------------------------------------
cl::Program MorphOpenCL::createProgram(const QString& progFile, 
	const QString& options)
{
	QByteArray b = progFile.toAscii();
	const char* pname = b.data();
	const char* oname = nullptr;

	if(!options.isEmpty())
	{
		QByteArray b1 = options.toAscii();
		oname = b1.data();
	}

	return createProgram(pname, oname);
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

		printf("Building %s program...", progFile);
		err = program.build(devs, options);
		QString log(QString::fromStdString(
			program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev)));
		if(err != CL_SUCCESS)
			clError(log, err);
		printf("[OK]\n");

		if(log.size() > 0)
			printf("log: %s\n", log.toAscii().constData());

		programs[progFile] = program;
		return program;
	}
	else
	{
		return it->second;
	}	
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCL::zeroAtomicCounter(const cl::Buffer& clAtomicCounter)
{
	// Zapisuje wartosc 0 do wskazanego bufora
	// (moze to byc zwykly bufor lub licznik atomowy)
	static cl_uint d_init = 0;
	cl::Event evt;
	cl_int err= cq.enqueueWriteBuffer(clAtomicCounter, CL_FALSE,
		0, 	sizeof(cl_uint), &d_init, nullptr, &evt);
	clError("Error while zeroing atomic counter value!", err);
	evt.wait();
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCL::readAtomicCounter(cl_uint& v, 
	const cl::Buffer& clAtomicCounter)
{
	// Odczytuje pojedyncza wartosc ze wskazanego bufora
	// (moze to byc zwykly bufor lub licznik atomowy)
	cl::Event evt;
	cl_int err = cq.enqueueReadBuffer(clAtomicCounter, CL_FALSE,
		0, sizeof(cl_uint), &v, nullptr, &evt);
	clError("Error while reading atomic counter value!", err);
	evt.wait();
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl::Kernel MorphOpenCL::createKernel(const cl::Program& prog, 
	const char* kernelName)
{
	cl_int err;
	printf("Creating %s kernel...", kernelName);
	cl::Kernel k(prog, kernelName, &err);
	clError("Failed to create " + 
		QString(kernelName) + " kernel!", err);
	printf("[OK]\n");
	return k;
}
// -------------------------------------------------------------------------
cl::Kernel MorphOpenCL::createKernel(const cl::Program& prog, 
	const QString& kernelName)
{
	QByteArray b = kernelName.toAscii();
	const char* kname = b.data();

	return createKernel(prog, kname);
}
// -------------------------------------------------------------------------
QString MorphOpenCL::openCLErrorCodeStr(cl_int errcode)
{
	switch(errcode)
	{
	case CL_SUCCESS: return "CL_SUCCESS";
	case CL_DEVICE_NOT_FOUND: return "CL_DEVICE_NOT_FOUND";
	case CL_DEVICE_NOT_AVAILABLE: return "CL_DEVICE_NOT_AVAILABLE";
	case CL_COMPILER_NOT_AVAILABLE: return "CL_COMPILER_NOT_AVAILABLE";
	case CL_MEM_OBJECT_ALLOCATION_FAILURE: return "CL_MEM_OBJECT_ALLOCATION_FAILURE";
	case CL_OUT_OF_RESOURCES: return "CL_OUT_OF_RESOURCES";
	case CL_OUT_OF_HOST_MEMORY: return "CL_OUT_OF_HOST_MEMORY";
	case CL_PROFILING_INFO_NOT_AVAILABLE: return "CL_PROFILING_INFO_NOT_AVAILABLE";
	case CL_MEM_COPY_OVERLAP: return "CL_MEM_COPY_OVERLAP";
	case CL_IMAGE_FORMAT_MISMATCH: return "CL_IMAGE_FORMAT_MISMATCH";
	case CL_IMAGE_FORMAT_NOT_SUPPORTED: return "CL_IMAGE_FORMAT_NOT_SUPPORTED";
	case CL_BUILD_PROGRAM_FAILURE: return "CL_BUILD_PROGRAM_FAILURE";
	case CL_MAP_FAILURE: return "CL_MAP_FAILURE";
	case CL_MISALIGNED_SUB_BUFFER_OFFSET: return "CL_MISALIGNED_SUB_BUFFER_OFFSET";
	case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST: return "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
	case CL_INVALID_VALUE: return "CL_INVALID_VALUE";
	case CL_INVALID_DEVICE_TYPE: return "CL_INVALID_DEVICE_TYPE";
	case CL_INVALID_PLATFORM: return "CL_INVALID_PLATFORM";
	case CL_INVALID_DEVICE: return "CL_INVALID_DEVICE";
	case CL_INVALID_CONTEXT: return "CL_INVALID_CONTEXT";
	case CL_INVALID_QUEUE_PROPERTIES: return "CL_INVALID_QUEUE_PROPERTIES";
	case CL_INVALID_COMMAND_QUEUE: return "CL_INVALID_COMMAND_QUEUE";
	case CL_INVALID_HOST_PTR: return "CL_INVALID_HOST_PTR";
	case CL_INVALID_MEM_OBJECT: return "CL_INVALID_MEM_OBJECT";
	case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR: return "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
	case CL_INVALID_IMAGE_SIZE: return "CL_INVALID_IMAGE_SIZE";
	case CL_INVALID_SAMPLER: return "CL_INVALID_SAMPLER";
	case CL_INVALID_BINARY: return "CL_INVALID_BINARY";
	case CL_INVALID_BUILD_OPTIONS: return "CL_INVALID_BUILD_OPTIONS";
	case CL_INVALID_PROGRAM: return "CL_INVALID_PROGRAM";
	case CL_INVALID_PROGRAM_EXECUTABLE: return "CL_INVALID_PROGRAM_EXECUTABLE";
	case CL_INVALID_KERNEL_NAME: return "CL_INVALID_KERNEL_NAME";
	case CL_INVALID_KERNEL_DEFINITION: return "CL_INVALID_KERNEL_DEFINITION";
	case CL_INVALID_KERNEL: return "CL_INVALID_KERNEL";
	case CL_INVALID_ARG_INDEX: return "CL_INVALID_ARG_INDEX";
	case CL_INVALID_ARG_VALUE: return "CL_INVALID_ARG_VALUE";
	case CL_INVALID_ARG_SIZE: return "CL_INVALID_ARG_SIZE";
	case CL_INVALID_KERNEL_ARGS: return "CL_INVALID_KERNEL_ARGS";
	case CL_INVALID_WORK_DIMENSION: return "CL_INVALID_WORK_DIMENSION";
	case CL_INVALID_WORK_GROUP_SIZE: return "CL_INVALID_WORK_GROUP_SIZE";
	case CL_INVALID_WORK_ITEM_SIZE: return "CL_INVALID_WORK_ITEM_SIZE";
	case CL_INVALID_GLOBAL_OFFSET: return "CL_INVALID_GLOBAL_OFFSET";
	case CL_INVALID_EVENT_WAIT_LIST: return "CL_INVALID_EVENT_WAIT_LIST";
	case CL_INVALID_EVENT: return "CL_INVALID_EVENT";
	case CL_INVALID_OPERATION: return "CL_INVALID_OPERATION";
	case CL_INVALID_GL_OBJECT: return "CL_INVALID_GL_OBJECT";
	case CL_INVALID_BUFFER_SIZE: return "CL_INVALID_BUFFER_SIZE";
	case CL_INVALID_MIP_LEVEL: return "CL_INVALID_MIP_LEVEL";
	case CL_INVALID_GLOBAL_WORK_SIZE: return "CL_INVALID_GLOBAL_WORK_SIZE";
	case CL_INVALID_PROPERTY: return "CL_INVALID_PROPERTY";
	default: return "UNKNOWN";
	}
}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// MorphOpenCLImage

bool MorphOpenCLImage::initOpenCL()
{
	MorphOpenCL::initOpenCL();

	// Pobierz obslugiwane formaty obrazow
	std::vector<cl::ImageFormat> imageFormats;
	context.getSupportedImageFormats(CL_MEM_READ_WRITE,
		CL_MEM_OBJECT_IMAGE2D, &imageFormats);

	// Sprawdz czy ten co chcemy jest obslugiwany
	bool found = false;
	imageFormat.image_channel_data_type = CL_UNSIGNED_INT8;
	imageFormat.image_channel_order = CL_R;

	for(auto i = imageFormats.cbegin(), 
		ie = imageFormats.cend(); i != ie; ++i)
	{
		if (i->image_channel_data_type == imageFormat.image_channel_data_type &&
			i->image_channel_order == imageFormat.image_channel_order)
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

	// Wczytaj programy
	cl::Program perode = createProgram("kernels-images/erode.cl");
	cl::Program pdilate = createProgram("kernels-images/dilate.cl");
	cl::Program pthinning = createProgram("kernels-images/thinning.cl");
	cl::Program putils = createProgram("kernels-images/utils.cl");
	cl::Program pskeleton = createProgram("kernels-images/skeleton.cl");

	QSettings s("./settings.cfg", QSettings::IniFormat);

	// Stworz kernele (nazwy pobierz z pliku konfiguracyjnego)
	kernelErode = createKernel(perode, s.value("kernel/erode", "erode").toString());
	kernelDilate = createKernel(pdilate, s.value("kernel/dilate", "dilate").toString());
	kernelThinning = createKernel(pthinning, s.value("kernel/thinning", "thinning").toString());
	kernelSubtract = createKernel(putils, s.value("kernel/subtract", "subtract").toString());

	for(int i = 0; i < 8; ++i)
	{
		QString kernelName = "skeleton_iter" + QString::number(i+1);
		kernelSkeleton_iter[i] = createKernel(pskeleton, kernelName);
	}

	return true;
}
// -------------------------------------------------------------------------
void MorphOpenCLImage::setSourceImage(const cv::Mat* newSrc)
{
	cl_int err;

	// Zaladuj obraz zrodlowy do karty
	clSrcImage = cl::Image2D(context,
		CL_MEM_READ_ONLY, imageFormat,
		newSrc->cols, newSrc->rows, 0, 
		nullptr, &err);
	clError("Error while creating OpenCL source image!", err);

	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;

	cl::size_t<3> region;
	region[0] = newSrc->cols;
	region[1] = newSrc->rows;
	region[2] = 1;

	cl::Event evt;
	err = cq.enqueueWriteImage(clSrcImage, CL_FALSE, 
		origin, region, 0, 0,
		const_cast<uchar*>(newSrc->ptr<uchar>()), 0, &evt);
	evt.wait();

	// Podaj czas trwania transferu
	cl_ulong delta = elapsedEvent(evt);
	printf("Transfering source image to GPU took %.3lf ms\n", delta * 0.000001);

	clError("Error while writing new data to OpenCL source image!", err);
	src = newSrc;
}
// -------------------------------------------------------------------------
double MorphOpenCLImage::morphology(EOperationType opType, cv::Mat& dst, int& iters)
{
	// Obraz docelowy
	cl_int err;
	clDstImage = cl::Image2D(context,
		CL_MEM_WRITE_ONLY, imageFormat,
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
		// Kopiuje caly obraz, zwraca czas operacji
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

		// Operacja wyciagania konturow
		if(opType == OT_Thinning)
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;
			elapsed += copyImage(clSrcImage, clDstImage, evt);

			// Wykonaj operacje hitmiss
			elapsed += executeHitMissKernel(&kernelThinning, clSrcImage, clDstImage);
		}
		else
		{
			// Potrzebowac bedziemy dodatkowego bufora tymczasowego
			clTmpImage = cl::Image2D(context,
				CL_MEM_READ_WRITE, imageFormat,
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
			// Operacja szkieletyzacji
			else if(opType == OT_Skeleton)
			{
				iters = 0;

				// Skopiuj obraz zrodlowy do docelowego i tymczasowego
				cl::Event evt;	
				elapsed += copyImage(clSrcImage, clTmpImage, evt);
				elapsed += copyImage(clSrcImage, clDstImage, evt);

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
							clTmpImage, clDstImage, &clAtomicCnt);

						// Kopiowanie obrazu
						elapsed += copyImage(clDstImage, clTmpImage, evt);
					}

					// Sprawdz ile pikseli zostalo zmodyfikowanych
					cl_uint diff;
					elapsed += readAtomicCounter(diff, clAtomicCnt);

					// Sprawdz warunek stopu
					if(diff == 0)
						break;

					elapsed += zeroAtomicCounter(clAtomicCnt);	

				} while(true);
			}
			else
			{
				// Potrzebowac bedziemy jeszcze jednego dodatkowego bufora tymczasowego
				clTmp2Image = cl::Image2D(context,
					CL_MEM_READ_WRITE, imageFormat,
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
					// Nieznana operacja - do tego miejsca nie powinnismy nigdy dojsc
					iters = 0;
					dst.create(src->size(), CV_8U);
					return 0.0;		
				}
			}
		}
	}
	
	// Zczytaj wynik z karty
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;

	// Chcemy caly obszar
	cl::size_t<3> region;
	region[0] = src->cols; 
	region[1] = src->rows;
	region[2] = 1;

	dst.create(src->size(), CV_8U);
	cl::Event evt;
	err = cq.enqueueReadImage(clDstImage, CL_FALSE, origin,
		region, 0, 0, dst.ptr<uchar>(),
		nullptr, &evt);
	clError("Error while reading result to image buffer!", err);
	evt.wait();

	// Ile czasu zajelo zczytanie danych z powrotem
	cl_ulong readingTime = elapsedEvent(evt);
	double totalTime = (elapsed + readingTime) * 0.000001;
	printf("Processing time: %.3lf ms (in which %.3lf ms was transfer time)\n",
		totalTime, readingTime * 0.000001);

	// Ile czasu wszystko zajelo
	return totalTime;
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

	//cl::NDRange offset(kradiusx, kradiusy);
	//cl::NDRange gridDim(src->cols - 2*kradiusx, src->rows - 2*kradiusy);

	cl::NDRange offset(0, 0);
	cl::NDRange gridDim(src->cols, src->rows);

	// Odpal kernela
	cl::Event evt;	
	err |= cq.enqueueNDRangeKernel(*kernel,
		offset, gridDim,
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLImage::executeHitMissKernel(cl::Kernel* kernel, 
	const cl::Image2D& clSrcImage, cl::Image2D& clDstImage,
	cl::Buffer* clAtomicCnt)
{
	// Ustaw argumenty kernela
	cl_int err;
	err  = kernel->setArg(0, clSrcImage);
	err |= kernel->setArg(1, clDstImage);
	if(clAtomicCnt)
		err |= kernel->setArg(2, *clAtomicCnt);

	clError("Error while setting kernel arguments", err);

	cl::NDRange offset(1, 1);
	cl::NDRange gridDim(src->cols - 2, src->rows - 2);

	// Odpal kernela
	cl::Event evt;
	err |= cq.enqueueNDRangeKernel(*kernel,
		offset, gridDim,
		cl::NullRange,
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

	// Odpal kernela
	cl::Event evt;	
	err |= cq.enqueueNDRangeKernel(kernelSubtract,
		cl::NullRange,
		cl::NDRange(src->cols, src->rows),
		cl::NullRange, 
		nullptr, &evt);
	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}

// HHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHHH
// MorphOpenCLBuffer

// -------------------------------------------------------------------------
bool MorphOpenCLBuffer::initOpenCL()
{
	MorphOpenCL::initOpenCL();

	QSettings s("./settings.cfg", QSettings::IniFormat);

	// Typ danych (uchar czy uint)
	QString dir;
	if(s.value("misc/datatype", "0").toInt() == 0)
	{
		dir = "kernels-buffers/";
		useUint = false;
	}
	else
	{
		dir = "kernels-uint/";
		useUint = true;
	}

	// Wczytaj programy
	cl::Program perode = createProgram(dir + "erode.cl");
	cl::Program pdilate = createProgram(dir + "dilate.cl");
	cl::Program pthinning = createProgram(dir + "thinning.cl");
	cl::Program putils = createProgram(dir + "utils.cl");
	cl::Program pskeleton = createProgram(dir + "skeleton.cl");	

	// Stworz kernele (nazwy pobierz z pliku konfiguracyjnego)
	kernelErode = createKernel(perode, s.value("kernel/erode", "erode").toString());
	kernelDilate = createKernel(pdilate, s.value("kernel/dilate", "dilate").toString());
	kernelThinning = createKernel(pthinning, s.value("kernel/thinning", "thinning").toString());
	kernelSubtract = createKernel(putils, s.value("kernel/subtract", "subtract").toString());

	// subtract4 (wymaga wyrownania wierszy danych do 4 bajtow) czy subtract
	QString sub = s.value("kernel/subtract", "subtract").toString();
	if(sub.endsWith("4")) sub4 = true;
	else sub4 = false;

	int local = s.value("kernel/local", "0").toInt();
	for(int i = 0; i < 8; ++i)
	{
		if(local == 0)
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

	return true;
}
// -------------------------------------------------------------------------
void MorphOpenCLBuffer::setSourceImage(const cv::Mat* newSrc)
{
	cl_int err;

	// Czy chcemy aby bufor mial wyrownane wiersze danych do rozmiary grupy roboczej
	if(readingMethod == RM_NotOptimized)
		deviceWidth = newSrc->cols;
	else
		deviceWidth = roundUp(newSrc->cols, workGroupSizeX);

	// Rozmiar bufora
	deviceHeight = newSrc->rows;
	int bufferDeviceSize = deviceWidth * deviceHeight;
	
	if(useUint)
		bufferDeviceSize *= sizeof(cl_uint);

	clSrc = cl::Buffer(context, CL_MEM_READ_ONLY, bufferDeviceSize, nullptr, &err);
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

	// Skopiuj dane 1:1
	if(readingMethod == RM_NotOptimized)
	{
		err = cq.enqueueWriteBuffer(clSrc, CL_TRUE, 0, 
			bufferDeviceSize, 
			srcptr, 0, &evt);
	}
	// Skopiuj dane tak by byly odpowiednio wyrownane
	else
	{
		cl::size_t<3> origin;
		origin[0] = 0;
		origin[1] = 0;
		origin[2] = 0;

		cl::size_t<3> region;
		region[0] = newSrc->cols;
		region[1] = newSrc->rows;
		region[2] = 1;

		size_t buffer_row_pitch = deviceWidth;
		size_t host_row_pitch = newSrc->cols;

		if(useUint)
		{
			region[0] *= sizeof(uint);

			buffer_row_pitch *= sizeof(uint);
			host_row_pitch *= sizeof(uint);
		}

		err = cq.enqueueWriteBufferRect(clSrc, CL_TRUE, 
			origin, origin, region, 
			buffer_row_pitch, 0, 
			host_row_pitch, 0, 
			srcptr, 0, &evt);
	}

	evt.wait();
	clError("Error while writing new data to OpenCL source buffer!", err);

	// Podaj czas trwania transferu
	cl_ulong delta = elapsedEvent(evt);
	printf("Transfering source image to GPU took %.3lfms\n", delta * 0.000001);

	delete [] ptr;	
	src = newSrc;
}
// -------------------------------------------------------------------------
double MorphOpenCLBuffer::morphology(EOperationType opType, cv::Mat& dst, int& iters)
{
	int dstSizeX = src->cols;
	int dstSizeY = src->rows;
	size_t bufferDeviceSize = deviceWidth * deviceHeight;

	if(useUint)
		bufferDeviceSize *= sizeof(cl_uint);

	// Bufor docelowy
	cl_int err;
	clDst = cl::Buffer(context,
		CL_MEM_WRITE_ONLY, 
		bufferDeviceSize, // obraz 1-kanalowy
		nullptr, &err);
	clError("Error while creating destination OpenCL buffer!", err);

	iters = 1;
	cl_ulong elapsed = 0;

	if(opType == OT_Erode)
	{
		elapsed += executeMorphologyKernel(&kernelErode, clSrc, clDst);
		dstSizeX -= kradiusx*2;
		dstSizeY -= kradiusy*2;
	}
	// Dylatacja
	else if(opType == OT_Dilate)
	{
		elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clDst);
		dstSizeX -= kradiusx*2;
		dstSizeY -= kradiusy*2;
	}
	else
	{
		// Funkcja lambda kopiujaca zawartosc jednego bufora OpenCL'a do drugiego
		// przy okazji mierzac czas tej operacji
		auto copyBuffer = [=](const cl::Buffer& clsrc, cl::Buffer& cldst,
			cl::Event& clevt) -> cl_ulong
		{
			cq.enqueueCopyBuffer(clsrc, cldst, 
				0, 0, bufferDeviceSize, 
				nullptr, &clevt);
			clevt.wait();
			return elapsedEvent(clevt);
		};

		// Operacja wyciagania konturow
		if(opType == OT_Thinning)
		{
			// Skopiuj obraz zrodlowy do docelowego
			cl::Event evt;
			elapsed += copyBuffer(clSrc, clDst, evt);

			// Wykonaj operacje hitmiss
			elapsed += executeHitMissKernel(&kernelThinning, clSrc, clDst);
			dstSizeX -= 2;
			dstSizeY -= 2;
		}
		else
		{
			// Potrzebowac bedziemy dodatkowego bufora tymczasowego
			clTmp = cl::Buffer(context,
				CL_MEM_READ_WRITE, 
				bufferDeviceSize, // obraz 1-kanalowy
				nullptr, &err);
			clError("Error while creating temporary OpenCL buffer", err);

			// Otwarcie
			if(opType == OT_Open)
			{
				// dst = dilate(erode(src))
				elapsed += executeMorphologyKernel(&kernelErode, clSrc, clTmp);
				dstSizeX -= kradiusx*2;
				dstSizeY -= kradiusy*2;

				elapsed += executeMorphologyKernel(&kernelDilate, clTmp, clDst);
				dstSizeX -= kradiusx*2;
				dstSizeY -= kradiusy*2;
			}
			// Zamkniecie
			else if(opType == OT_Close)
			{
				// dst = erode(dilate(src))
				elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clTmp);
				dstSizeX -= kradiusx*2;
				dstSizeY -= kradiusy*2;

				elapsed += executeMorphologyKernel(&kernelErode, clTmp, clDst);
				dstSizeX -= kradiusx*2;
				dstSizeY -= kradiusy*2;
			}
			// Operacja szkieletyzacji
			else if(opType == OT_Skeleton)
			{
				// Operacja szkieletyzacji
				if(opType == OT_Skeleton)
				{
					iters = 0;

					// Skopiuj obraz zrodlowy do docelowego i tymczasowego
					cl::Event evt;	
					elapsed += copyBuffer(clSrc, clTmp, evt);
					elapsed += copyBuffer(clSrc, clDst, evt);				

					// Licznik atomowy (ew. zwyczajny bufor)
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
								clTmp, clDst, &clAtomicCnt);

							// Kopiowanie bufora
							elapsed += copyBuffer(clDst, clTmp, evt);
						}

						// Sprawdz ile pikseli zostalo zmodyfikowanych
						cl_uint diff;
						elapsed += readAtomicCounter(diff, clAtomicCnt);

						// Sprawdz warunek stopu
						if(diff == 0)
							break;

						elapsed += zeroAtomicCounter(clAtomicCnt);

					} while (true);

					dstSizeX -= 2;
					dstSizeY -= 2;
				}
			}
			else
			{
				// Potrzebowac bedziemy jeszcze jednego dodatkowego bufora tymczasowego
				clTmp2 = cl::Buffer(context,
					CL_MEM_READ_WRITE, 
					bufferDeviceSize, // obraz 1-kanalowy
					nullptr, &err);
				clError("Error while creating temporary OpenCL buffer", err);

				// Gradient morfologiczny
				if(opType == OT_Gradient)
				{ 
					//dst = dilate(src) - erode(src);
					elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clTmp);
					elapsed += executeMorphologyKernel(&kernelErode, clSrc, clTmp2);
					elapsed += executeSubtractKernel(clTmp, clTmp2, clDst);

					dstSizeX -= kradiusx*2;
					dstSizeY -= kradiusy*2;
				}
				// TopHat
				else if(opType == OT_TopHat)
				{ 
					// dst = src - dilate(erode(src))
					elapsed += executeMorphologyKernel(&kernelErode, clSrc, clTmp);
					dstSizeX -= kradiusx*2;
					dstSizeY -= kradiusy*2;

					elapsed += executeMorphologyKernel(&kernelDilate, clTmp, clTmp2);
					dstSizeX -= kradiusx*2;
					dstSizeY -= kradiusy*2;

					elapsed += executeSubtractKernel(clSrc, clTmp2, clDst);
				}
				// BlackHat
				else if(opType == OT_BlackHat)
				{ 
					// dst = close(src) - src
					elapsed += executeMorphologyKernel(&kernelDilate, clSrc, clTmp);
					dstSizeX -= kradiusx*2;
					dstSizeY -= kradiusy*2;

					elapsed += executeMorphologyKernel(&kernelErode, clTmp, clTmp2);
					dstSizeX -= kradiusx*2;
					dstSizeY -= kradiusy*2;

					elapsed += executeSubtractKernel(clTmp2, clSrc, clDst);
				}
				else
				{
					// Nieznana operacja - do tego miejsca nie powinnismy nigdy dojsc
					iters = 0;
					dst.create(src->size(), CV_8U);
					return 0.0;				
				}
			}
		}
	}

	// Zczytaj wynik z karty
	//dst.create(src->size(), CV_8U);

	// Czasami zostaja smieci
	dst = cv::Mat(src->size(), CV_8U, cv::Scalar(0));
	cl::Event evt;

	cl::size_t<3> origin;
	cl::size_t<3> region;

	origin[0] = (src->cols - dstSizeX)/2;
	origin[1] = (src->rows - dstSizeY)/2;
	origin[2] = 0;

	region[0] = dstSizeX;
	region[1] = dstSizeY;
	region[2] = 1;

	size_t buffer_row_pitch = deviceWidth;
	size_t host_row_pitch = src->cols;

	if(useUint)
	{
		// Musimy wczytac wiecej danych
		origin[0] *= sizeof(uint);
		region[0] *= sizeof(uint);

		buffer_row_pitch *= sizeof(uint);
		host_row_pitch *= sizeof(uint);

		// .. do tymczasowego bufora uint'ow
		uint* dstTmp = new uint[src->size().area()];

		err = cq.enqueueReadBufferRect(clDst, CL_FALSE, 
			origin, origin, region, 
			buffer_row_pitch, 0, 
			host_row_pitch, 0, 
			dstTmp, nullptr, &evt);

		clError("Error while reading result to buffer!", err);
		evt.wait();

		// .. a nastepnie zrzutowac do uchar'ow
		uchar* dptr = dst.ptr<uchar>();
		for(int i = 0; i < dst.cols * dst.rows; ++i)
			//dptr[i] = cv::saturate_cast<uchar>(dstTmp[i]);
			dptr[i] = static_cast<uchar>(dstTmp[i]);

		delete [] dstTmp;
	}
	else
	{
		err = cq.enqueueReadBufferRect(clDst, CL_FALSE, 
			origin, origin, region, 
			buffer_row_pitch, 0, 
			host_row_pitch, 0, 
			dst.ptr<uchar>(), nullptr, &evt);

		clError("Error while reading result to buffer!", err);
		evt.wait();
	}

	// Ile czasu zajelo zczytanie danych z powrotem
	cl_ulong readingTime = elapsedEvent(evt);
	double totalTime = (elapsed + readingTime) * 0.000001;
	printf("Processing time: %.3lf ms (in which %.3lf ms was transfer time)\n",
		totalTime, readingTime * 0.000001);

	// Ile czasu wszystko zajelo
	return totalTime;
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::executeMorphologyKernel(cl::Kernel* kernel, 
	const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer)
{
	cl::Event evt;
	cl_int err;

	cl_int4 seSize = { kradiusx, kradiusy, (int)(csize), 0 };
	cl_int2 imageSize = { deviceWidth, deviceHeight };

	if(!local)
	{
		// Ustaw argumenty kernela
		err  = kernel->setArg(0, clSrcBuffer);
		err |= kernel->setArg(1, clDstBuffer);
		err |= kernel->setArg(2, clSeCoords);
		err |= kernel->setArg(3, seSize);
		err |= kernel->setArg(4, imageSize);
		clError("Error while setting kernel arguments", err);

		// Odpal kernela
		err = cq.enqueueNDRangeKernel(*kernel,
			cl::NullRange,
			cl::NDRange(src->cols - kradiusx*2, src->rows - kradiusy*2),
			cl::NullRange, 
			nullptr, &evt);
	}
	else
	{
		int apronX = kradiusx * 2;
		int apronY = kradiusy * 2;

		int globalItemsX = roundUp(src->cols - apronX, workGroupSizeX);
		int globalItemsY = roundUp(src->rows - apronY, workGroupSizeX);	

		cl_int2 sharedSize = {
			roundUp(workGroupSizeX + apronX, 4),
			workGroupSizeY + apronY
		};
		size_t sharedBlockSize = sharedSize.s[0] * sharedSize.s[1];
		if(useUint) sharedBlockSize *= sizeof(cl_uint);

		// Ustaw argumenty kernela
		err  = kernel->setArg(0, clSrcBuffer);
		err |= kernel->setArg(1, clDstBuffer);
		err |= kernel->setArg(2, clSeCoords);
		err |= kernel->setArg(3, seSize);
		err |= kernel->setArg(4, imageSize);
		err |= kernel->setArg(5, sharedBlockSize, nullptr);
		err |= kernel->setArg(6, sharedSize);
		clError("Error while setting kernel arguments", err);

		// Odpal kernela
		err = cq.enqueueNDRangeKernel(*kernel,
			cl::NullRange,
			cl::NDRange(globalItemsX, globalItemsY),
			cl::NDRange(workGroupSizeX, workGroupSizeY), 
			nullptr, &evt);
	}

	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
// -------------------------------------------------------------------------
cl_ulong MorphOpenCLBuffer::executeHitMissKernel(cl::Kernel* kernel, 
	const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer, 
	cl::Buffer* clAtomicCounter)
{
	cl::Event evt;
	cl_int err;

	if(!local)
	{
		// Ustaw argumenty kernela
		err  = kernel->setArg(0, clSrcBuffer);
		err |= kernel->setArg(1, clDstBuffer);
		err |= kernel->setArg(2, deviceWidth);
		if(clAtomicCounter)
			err |= kernel->setArg(3, *clAtomicCounter);

		clError("Error while setting kernel arguments", err);

		// Odpal kernela
		err = cq.enqueueNDRangeKernel(*kernel,
			cl::NDRange(1, 1),
			cl::NDRange(src->cols - 2, src->rows - 2),
			cl::NullRange,
			nullptr, &evt);
	}
	else
	{
		cl_int2 imageSize = { deviceWidth, deviceHeight };
		const int lsize = 16;

		// Ustaw argumenty kernela
		err  = kernel->setArg(0, clSrcBuffer);
		err |= kernel->setArg(1, clDstBuffer);
		err |= kernel->setArg(2, imageSize);
		if(clAtomicCounter)
			err |= kernel->setArg(3, *clAtomicCounter);

		clError("Error while setting kernel arguments", err);

		// Odpal kernela
		err = cq.enqueueNDRangeKernel(*kernel,
			cl::NullRange,
			cl::NDRange(roundUp(src->cols - 2, lsize), roundUp(src->rows - 2, lsize)),
			cl::NDRange(lsize, lsize), 
			nullptr, &evt);
	}

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

	// Ustaw argumenty kernela
	err  = kernelSubtract.setArg(0, clABuffer);
	err |= kernelSubtract.setArg(1, clBBuffer);
	err |= kernelSubtract.setArg(2, clDstBuffer);
	clError("Error while setting kernel arguments", err);

	int xitems = deviceWidth;
	if(sub4) xitems /= 4;

	// Odpal kernela
	err = cq.enqueueNDRangeKernel(kernelSubtract,
		cl::NullRange,
		cl::NDRange(xitems * deviceHeight),
		cl::NullRange, 
		nullptr, &evt);

	evt.wait();
	clError("Error while executing kernel over ND range!", err);

	// Ile czasu to zajelo
	return elapsedEvent(evt);
}
