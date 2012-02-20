#include "morphocl.h"
#include "morphoclimage.h"

#include <QFile>
#include <QTextStream>
#include <QSettings>

int roundUp(int value, int multiple)
{
	int v = value % multiple;
	if (v)
		return value + (multiple - v);
	return value;
}

MorphOpenCL::MorphOpenCL()
: error(false),
errorCallback(nullptr),
kradiusx(0),
kradiusy(0),
sharedw(-1), 
sharedh(-1),
useShared(false)
{
	// Wczytaj opcje z pliku konfiguracyjnego
	QSettings settings("./settings.cfg", QSettings::IniFormat);
	workGroupSizeX = settings.value("opencl/workgroupsizex", 16).toInt();
	workGroupSizeY = settings.value("opencl/workgroupsizey", 16).toInt();
	useShared      = settings.value("opencl/glinterop", "true").toBool();
}
// -------------------------------------------------------------------------
bool MorphOpenCL::initOpenCL()
{
	// Connect to a compute device
	std::vector<cl::Platform> platforms;
	cl_int err;
	err = cl::Platform::get(&platforms);
	if(platforms.empty())
	{
		error = true;
		printf("No OpenCL platfrom has been detected!");
		return false;
	}

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

		printf("\n");
		int choice = 0;
		while(choice > (int)(platforms.size()) || choice <= 0)
		{
			printf("Choose OpenCL platform: ");
			int r = scanf("%d", &choice);
			// Jesli nie odczytano jednej liczby (np. wprowadzono znak A)
			// trzeba oproznic stdin, inaczej wpadniemy w nieskonczona petle
			if(r != 1)
			{
				char buf[128];
				fgets(buf, 128, stdin);
			}
		}
		platform = platforms[choice-1];
	}

	cl_context_properties properties_interop[] = { 
		CL_CONTEXT_PLATFORM, (cl_context_properties) (platform)(),
		CL_GL_CONTEXT_KHR,   (cl_context_properties) wglGetCurrentContext(),
		CL_WGL_HDC_KHR,      (cl_context_properties) wglGetCurrentDC(),
		0, 0
	};	

	cl_context_properties properties[] = { 
		CL_CONTEXT_PLATFORM, (cl_context_properties) (platform)(),
		0, 0
	};

	// Stworz kontekst
	context = cl::Context(CL_DEVICE_TYPE_ALL,
		useShared ? properties_interop : properties, 
		nullptr, nullptr, &err);
	clError("Failed to create compute context!", err);
	if(error) return false;

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
		printf("\n");
		while(choice > (int)(devices.size()) || choice <= 0)
		{
			printf("Choose OpenCL device: ");
			int r = scanf("%d", &choice);
			// Jesli nie odczytano jednej liczby (np. wprowadzono znak A)
			// trzeba opronznic stdin, inaczej wpadniemy w nieskonczona petle
			if(r != 1)
			{
				char buf[128];
				fgets(buf, 128, stdin);
			}
		}
		dev = devices[choice-1];
	}

	std::vector<cl::Device> devs(1);
	devs[0] = (dev);

	cl_uint maxComputeUnits, maxClockFreq;
	cl_ulong maxMemAllocSize, localMemSize;
	cl_device_local_mem_type localMemType;

	printf("\n********************************************************\n");
	dev.getInfo(CL_DEVICE_MAX_COMPUTE_UNITS, &maxComputeUnits);
	printf("Compute units: %d\n", maxComputeUnits);

	dev.getInfo(CL_DEVICE_MAX_CLOCK_FREQUENCY, &maxClockFreq);
	printf("Clock frequency: %d Hz\n", maxClockFreq);

	dev.getInfo(CL_DEVICE_MAX_MEM_ALLOC_SIZE, &maxMemAllocSize);
	printf("Memory object allocation: %ld B (%ld MB)\n", maxMemAllocSize, maxMemAllocSize/1024/1024);

	dev.getInfo(CL_DEVICE_LOCAL_MEM_SIZE, &localMemSize);
	printf("Local memory arena: %ld B (%ld kB)\n", localMemSize, localMemSize/1024);

	dev.getInfo(CL_DEVICE_LOCAL_MEM_TYPE, &localMemType);
	printf("Local memory type: %s\n", ((localMemType == CL_LOCAL) ? "local":"global"));

	dev.getInfo(CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE, &maxConstantBufferSize);
	printf("Constant buffer size: %ld B (%ld kB)\n", maxConstantBufferSize,
		maxConstantBufferSize/1024);
	
	printf("********************************************************\n\n");

	// Kolejka polecen
	cq = cl::CommandQueue(context, dev, CL_QUEUE_PROFILING_ENABLE, &err);
	clError("Failed to create command queue!", err);
	if(error) return false;

	return true;
}
// -------------------------------------------------------------------------
int MorphOpenCL::setStructuringElement(const cv::Mat& selement)
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

			cl_int2 c = {{x, y}};
			if(shiftCoords)
			{ 
				c.s[0] -= kradiusx;
				c.s[1] -= kradiusy;
			}

			coords.push_back(c);
		}
	}
	csize = static_cast<int>(coords.size());
	printf("Structuring element size (number of 'white' pixels): %d (%dx%d) - %d B\n",
		csize, 2*kradiusx+1, 2*kradiusy+1, sizeof(cl_int2) * csize);

	if(maxConstantBufferSize < sizeof(cl_int2)*csize)
	{
		clError(QString("Structuring element is too big: "
			"%1 B out of available %2 B.")
				.arg(sizeof(cl_int2)*csize)
				.arg(maxConstantBufferSize), CL_OUT_OF_RESOURCES);
		return 0;
	}

	// Zaladuj dane do bufora
	cl_int err;
	clStructuringElementCoords = cl::Buffer(context,
		CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR,
		csize * sizeof(cl_int2), coords.data(), &err);
	clError("Error while creating buffer for structuring element!", err);

	return csize;
}
// -------------------------------------------------------------------------
void MorphOpenCL::clError(const QString& message, cl_int err)
{
	if(err != CL_SUCCESS)
	{
		error = true;
		if(errorCallback != nullptr)
			errorCallback(message, err);
	}
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
	std::string b1, b = progFile.toStdString();
	const char* oname = nullptr;

	if(!options.isEmpty())
	{
		b1 = options.toStdString();
		oname = b1.c_str();
	}

	return createProgram(b.c_str(), oname);
}
// -------------------------------------------------------------------------
cl::Program MorphOpenCL::createProgram(const char* progFile, const char* options)
{
	if(error) return cl::Program();

	QFile file(progFile);
	if(!file.open(QIODevice::ReadOnly | QIODevice::Text))
	{
		clError("Can't read " + 
			QString(progFile) + 
			" file!", -1);
		return cl::Program();
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
	if(error) return cl::Program();

	std::vector<cl::Device> devs(1);
	devs[0] = (dev);

	printf("Building %s program...", progFile);
	err = program.build(devs, options);
	QString log(QString::fromStdString(
		program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(dev)));
	if(err != CL_SUCCESS)
		clError(log, err);
	if(!error) printf("[OK]\n");

	if(log.size() > 0)
	{
		std::string slog = log.toStdString();
		printf("log: %s\n", slog.c_str());
	}
	return program;
}
// -------------------------------------------------------------------------
void MorphOpenCL::recompile(EOperationType opType, int coordsSize)
{
	static int prevCoordsSize[OT_Gradient+1] = {0};
	SKernelParameters* kparams;
	cl::Kernel* kernel;

	if(opType == OT_Erode)
	{
		kparams = &erodeParams;
		kernel = &kernelErode;
	}
	else if(opType == OT_Dilate)
	{
		kparams = &dilateParams;
		kernel = &kernelDilate;
	}
	else if(opType == OT_Gradient)
	{
		kparams = &gradientParams;
		kernel = &kernelGradient;
	}
	else
	{
		if(opType == OT_TopHat ||
		   opType == OT_BlackHat ||
		   opType == OT_Open ||
		   opType == OT_Close)
		{
			recompile(OT_Erode, coordsSize);
			recompile(OT_Dilate, coordsSize);
		}
		return;
	}

	if(!kparams->needRecompile || coordsSize == prevCoordsSize[opType])
		return;

	QString opts = kparams->options + " -DCOORDS_SIZE=" + QString::number(coordsSize);
	prevCoordsSize[opType] = coordsSize;

	cl::Program prog = createProgram(kparams->programName,opts);
	*kernel = createKernel(prog, kparams->kernelName);
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
	if(error) return cl::Kernel();

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
	std::string b = kernelName.toStdString();
	return createKernel(prog, b.c_str());
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
