#include "oclcontext.h"

#include <fstream>
#define CV_NO_BACKWARD_COMPATIBILITY
#include <opencv2/core/core.hpp>

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

oclContext::ErrorCallback oclContext::cb = nullptr;

oclContext::oclContext()
	: blockDim(8, 8)
	, retrievedPlatforms(false)
{

}

oclBufferHolder oclContext::copyDataToDevice(
	const void* data, size_t size,
	oclMemoryAccess access, bool async)
{
	// Zaladuj dane do bufora
	cl_int err;
	oclBufferHolder holder;

	holder.size = size;
	holder.buf = cl::Buffer(ctx,
		access, size, nullptr, &err);

	err = cq.enqueueWriteBuffer(holder.buf,
		async ? CL_TRUE : CL_FALSE,
		0, size, data, nullptr, &holder.evt);

	if(!async)
		holder.evt.wait();

	oclError("Error while creating buffer for structuring element!", err);
	return holder;
}

oclImage2DHolder oclContext::copyImageToDevice(
	const cv::Mat& image,
	oclMemoryAccess access, bool async)
{
	cl_int err;
	oclImage2DHolder holder;

	// na razie na stale
	static cl::ImageFormat format(CL_R, CL_UNORM_INT8);

	//holder.cpuImage = newSrc;
	holder.width = image.cols;
	holder.height = image.rows;
	holder.format = format;

	// Utworz nowy obraz o podanych parametrach
	holder.img = cl::Image2D(ctx,
		access, holder.format,
		image.cols, image.rows, 0,
		nullptr, &err);

	if(!oclError("Error while creating OpenCL source image!", err))
		return holder;

	// Zaladuj obraz zrodlowy do karty
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;

	cl::size_t<3> region;
	region[0] = image.cols;
	region[1] = image.rows;
	region[2] = 1;

	err = cq.enqueueWriteImage(holder.img,
		async ? CL_TRUE : CL_FALSE,
		origin, region, 0, 0,
		const_cast<uchar*>(image.ptr<uchar>()),
		0, &holder.evt);

	if(!async)
		holder.evt.wait();

	oclError("Error while writing new data to OpenCL image!", err);
	return holder;
}

cv::Mat oclContext::readImageFromDevice(
	const oclImage2DHolder& holder, bool async)
{
	// TODO
	// use holder.format
	int format= CV_8U;

	cv::Mat dst(cv::Size(holder.width, holder.height),
		format, cv::Scalar(0));

	// Zczytaj wynik z karty
	cl::size_t<3> origin;
	origin[0] = 0;
	origin[1] = 0;
	origin[2] = 0;

	// Chcemy caly obszar
	cl::size_t<3> region;
	region[0] = dst.cols;
	region[1] = dst.rows;
	region[2] = 1;

	cl_int err = cq.enqueueReadImage(holder.img,
		async ? CL_TRUE : CL_FALSE,
		origin, region, 0, 0,
		dst.ptr<uchar>(), nullptr,
		const_cast<cl::Event*>(&holder.evt));

	if(!async)
		holder.evt.wait();

	oclError("Error while reading result to image buffer!", err);
	return dst;

	// return elapsedEvent(evt);
}

oclImage2DHolder oclContext::createDeviceImage(
	int width, int height,
	oclMemoryAccess access)
{
	cl_int err;
	static cl::ImageFormat format(CL_R, CL_UNORM_INT8);

	oclImage2DHolder holder;
	holder.width = width;
	holder.height = height;
	holder.format = format;

	holder.img = cl::Image2D(ctx,
		access, holder.format,
		holder.width, holder.height,
		0, nullptr, &err);

	oclError("Error while creating OpenCL image2D.", err);
	return holder;
}

oclBufferHolder oclContext::createDeviceBuffer(
	int size, oclMemoryAccess access)
{
	cl_int err;
	oclBufferHolder holder;
	holder.size = size;

	holder.buf = cl::Buffer(ctx,
		access, size, nullptr, &err);

	oclError("Error while creating buffer for structuring element!", err);
	return holder;
}

void oclContext::copyDeviceImage(
	const oclImage2DHolder& src,
	oclImage2DHolder& dst, bool async)
{
	cl::size_t<3> origin;
	origin[0] = origin[1] = origin[2] = 0;

	cl::size_t<3> region;
	region[0] = dst.width;
	region[1] = dst.height;
	region[2] = 1;

	cl_int err = cq.enqueueCopyImage(src.img,
		dst.img,
		origin, origin, region,
		nullptr, &dst.evt);

	if(!async)
		dst.evt.wait();

	oclError("Error while copying one image to another", err);
	//return elapsedEvent(evt);
}

bool oclContext::retrievePlatforms(std::vector<oclPlatformDesc>& out)
{
	cl_int err = CL_SUCCESS;

	out.clear();
	if(!retrievedPlatforms)
		err = cl::Platform::get(&pls);

	if(pls.empty())
	{
		printf("No OpenCL platfrom has been detected!");
		return false;
	}

	for(size_t i = 0; i < pls.size(); ++i)
	{
		oclPlatformDesc newPl = {
			static_cast<int>(i),
			pls[i].getInfo<CL_PLATFORM_NAME>(&err),
			pls[i].getInfo<CL_PLATFORM_VERSION>(&err)
		};
		out.push_back(newPl);

		oclError("Error during retrieving platform info", err);
	}

	retrievedPlatforms = true;
	return true;
}

bool oclContext::createContext()
{
	cl_int err;
	ctx = cl::Context(CL_DEVICE_TYPE_ALL,
		nullptr, nullptr, nullptr, &err);

	return oclError("Error during creating default context", err);
}

bool oclContext::createContext(size_t platformId)
{
	if(!retrievedPlatforms)
		retrievePlatforms();

	if(platformId >= pls.size())
		return false;

	cl_context_properties properties[] = {
		CL_CONTEXT_PLATFORM, (cl_context_properties) (pls[platformId])(),
		0, 0
	};

	cl_int err;
	ctx = cl::Context(CL_DEVICE_TYPE_ALL,
		properties, nullptr, nullptr, &err);

	return oclError("Error during creating context", err);
}

bool oclContext::retrieveDevices(std::vector<oclDeviceDesc>& out)
{
	cl_int err;
	std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>(&err);
	if(!oclError("Couldn't retrieve list of OpenCL Devices", err))
		return false;

	for(size_t i = 0; i < devices.size(); ++i)
		out.emplace_back(populateDescription(devices[i]));

	if(!err)
		device = devices[0];

	return err == CL_SUCCESS;
}

void oclContext::chooseDevice(size_t deviceId)
{
	cl_int err;
	std::vector<cl::Device> devices = ctx.getInfo<CL_CONTEXT_DEVICES>(&err);

	if(deviceId < devices.size())
	{
		device = devices[deviceId];
		devDesc = populateDescription(device);
	}
}

bool oclContext::createCommandQueue(bool profiling)
{
	cl_int err;
	cq = cl::CommandQueue(ctx, device,
		profiling ? CL_QUEUE_PROFILING_ENABLE : 0, &err);
	return oclError("Error during creating command queue", err);
}

cl::Program oclContext::createProgram(const char* progFile,
	const char* options, bool forceBuild)
{
	const char* opts = options ? options : "";

	// Sprawdz czy program nie zostal juz zbudowany
	if(!forceBuild)
	{
		auto pr = programs.find(progFile);
		if(pr != programs.end())
		{
			auto propts = pr->second.find(opts);
			if(propts != pr->second.end())
			{
				printf("Using cached ocl program (opts: %s).\n", opts);
				return propts->second;
			}
		}
	}
	std::string src;
	if(!fileContents(progFile, src))
		return cl::Program();

	// Zbuduj program na podstawie podanego pliku zrodlowego oraz argumentow kompilacji
	cl_int err;
	cl::Program::Sources sources(1, std::make_pair(src.c_str(), src.size()));
	cl::Program program = cl::Program(ctx, sources, &err);
	if(!oclError("Can't create a program", err))
		return cl::Program();

	std::vector<cl::Device> devs(1);
	devs[0] = (device);

	printf("Building %s program (opts: %s)...", progFile, opts);

	err = program.build(devs, opts);
	std::string log(program.getBuildInfo<CL_PROGRAM_BUILD_LOG>(device));
	if(!oclError("Error during building a program", err))
	{
		if(log.size() > 0)
			printf("log: %s\n", log.c_str());
		return cl::Program();
	}

	// Kompilacja przebiegla pomyslnie
	printf("[OK]\n");
	if(log.size() > 0)
		printf("log: %s\n", log.c_str());

	// Dodaj zbudowany program do naszej 'biblioteki'
	auto pr = programs.find(progFile);
	if(pr != programs.end())
	{
		pr->second[opts] = program;
	}
	else
	{
		std::map<std::string, cl::Program> map;
		map[opts] = program;
		programs[progFile] = map;
	}

	return program;
}

cl::Kernel oclContext::retrieveKernel(
	const cl::Program& program,
	const char* kernelName)
{
	cl_int err;
	printf("Creating %s kernel...", kernelName);
	cl::Kernel kernel(program, kernelName, &err);
	if(!oclError("Failed to create kernel", err))
		return cl::Kernel();

	printf("[OK]\n");
	return kernel;
}

std::string oclContext::oclErrorString(cl_int code)
{
	switch(code)
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

bool oclContext::oclError(const std::string& message, cl_int err)
{
	if(err != CL_SUCCESS)
	{
		if(oclContext::cb != nullptr)
			oclContext::cb(message, err);
	}

	return !err;
}

double oclContext::oclElapsedEvent(const cl::Event& evt)
{
	cl_ulong eventstart = evt.getProfilingInfo<CL_PROFILING_COMMAND_START>();
	cl_ulong eventend = evt.getProfilingInfo<CL_PROFILING_COMMAND_END>();
	return static_cast<double>(eventend - eventstart) * 0.000001;
}

int oclContext::roundUp(int val, int multiple)
{
	int v = val % multiple;
	if (v)
		return val + (multiple - v);
	return val;
}

bool oclContext::retrievePlatforms()
{
	cl_int err = cl::Platform::get(&pls);

	if(pls.empty() || err != CL_SUCCESS)
	{
		printf("No OpenCL platfrom has been detected!");
		return false;
	}

	retrievedPlatforms = true;
	return true;
}

bool oclContext::fileContents(const char* progFile, std::string& ret)
{
	std::ifstream strm;
	strm.open(progFile, std::ios::binary | std::ios_base::in);
	if(!strm.is_open() || strm.bad())
	{
		printf("Error opening %s file.\n", progFile);
		return false;
	}

	// Alokuj miejscie na kod zrodlowy
	strm.seekg(0, std::ios::end);
	int len = static_cast<int>(strm.tellg());
	ret.reserve(len);
	strm.seekg(0);

	// Przepisz dane z pliku do tablicy
	ret.assign(std::istreambuf_iterator<char>(strm),
		std::istreambuf_iterator<char>());
	return true;
}

oclDeviceDesc oclContext::populateDescription(cl::Device dev)
{
	cl_int err;
	oclDeviceDesc desc = {
		dev.getInfo<CL_DEVICE_NAME>(&err),
		dev.getInfo<CL_DEVICE_IMAGE_SUPPORT>(&err) ? true : false,
		dev.getInfo<CL_DEVICE_TYPE>(&err),
		dev.getInfo<CL_DEVICE_MAX_COMPUTE_UNITS>(&err),
		dev.getInfo<CL_DEVICE_MAX_CLOCK_FREQUENCY>(&err),
		dev.getInfo<CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE>(&err),
		dev.getInfo<CL_DEVICE_MAX_MEM_ALLOC_SIZE>(&err),
		dev.getInfo<CL_DEVICE_LOCAL_MEM_SIZE>(&err),
		dev.getInfo<CL_DEVICE_LOCAL_MEM_TYPE>(&err)
	};

	oclError("Error during retrieving device info", err);
	return desc;
}