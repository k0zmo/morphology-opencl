#pragma once

#include <map>
#include <string>
#include <vector>

#include <CL/cl.hpp>
#include <functional>

#ifdef _MSC_VER
#define snprintf _snprintf
#endif

namespace cv {
class Mat;
}

enum oclMemoryAccess
{
	ReadWrite = CL_MEM_READ_WRITE,
	WriteOnly = CL_MEM_WRITE_ONLY,
	ReadOnly  = CL_MEM_READ_ONLY
};

struct oclPlatformDesc
{
	int id;
	std::string name;
	std::string version;
};

struct oclDeviceDesc
{
	std::string name;
	bool imagesSupported;
	cl_device_type deviceType;
	cl_uint maxComputeUnits;
	cl_uint maxClockFreq;
	cl_ulong maxConstantBufferSize;
	cl_ulong maxMemAllocSize;
	cl_ulong localMemSize;
	cl_device_local_mem_type localMemType;
};

struct oclBufferHolder
{
	int size;
	cl::Buffer buf;
	cl::Event evt;
};

struct oclBuffer2DHolder
{
	int width;
	int height;
	int step;
	cl::Buffer buf;
	cl::Event evt;
};

struct oclImage2DHolder
{
	int width;
	int height;
	cl::Image2D img;
	cl::Event evt;
	cl::ImageFormat format;
};

class oclContext
{
public:
	oclContext();

	bool retrievePlatforms(std::vector<oclPlatformDesc>& out);
	bool createContext();
	bool createContext(size_t platformId);
	bool retrieveDevices(std::vector<oclDeviceDesc>& out);

	void chooseDevice(size_t deviceId);
	bool createCommandQueue(bool profiling);

	cl::Program createProgram(const char* progFile,
		const char* options, bool forceBuild = false);
	cl::Kernel retrieveKernel(const cl::Program& program,
		const char* kernelName);	

	// TODO gdzie to dac?
	cl::NDRange blockDim;
	void setWorkgroupSize(int x, int y)
	{ blockDim = cl::NDRange(x, y); }
	cl::NDRange workgroupSize() const
	{ return blockDim; }

	// cpu data -> device buffer
	oclBufferHolder copyDataToDevice(const void* data, size_t size,
		oclMemoryAccess access, bool async = false);

	// cpu image -> device image
	oclImage2DHolder copyImageToDevice(const cv::Mat& image,
		oclMemoryAccess access, bool async = false);

	// device image -> cpu image
	cv::Mat readImageFromDevice(const oclImage2DHolder& holder, bool async = false);

	// Create empty device image
	oclImage2DHolder createDeviceImage(int width, int height,
		oclMemoryAccess access);

	// Create empty device buffer
	oclBufferHolder createDeviceBuffer(int size, oclMemoryAccess access);

	// Copies one device image to another without host involved
	void copyDeviceImage(const oclImage2DHolder& src,
		oclImage2DHolder& dst, bool async = false);

	// Zapisuje wartosc 0 do wskazanego bufora
	// (moze to byc zwykly bufor lub licznik atomowy)	
	template<typename AtomType>
	void zeroAtomicCounter(oclBufferHolder& counter,
		bool async = false);

	// Odczytuje pojedyncza wartosc ze wskazanego bufora
	// (moze to byc zwykly bufor lub licznik atomowy)
	template<typename AtomType>
	void readAtomicCounter(const oclBufferHolder& counter,
		AtomType* value, bool async = false);
	
	// Pomocznicze funkcje
	static std::string oclErrorString(cl_int code);
	static bool oclError(const std::string& message, cl_int err);
	static double oclElapsedEvent(const cl::Event& evt);
	static int roundUp(int val, int multiple);
	
	typedef std::function<void(const std::string&, cl_int)> ErrorCallback;
	static ErrorCallback cb;

	cl::Context context() const { return ctx; }
	cl::CommandQueue commandQueue() const { return cq; }
	oclDeviceDesc deviceDescription() { return devDesc; }

private:
	std::vector<cl::Platform> pls;
	std::map<std::string, std::map<std::string, cl::Program>> programs;
	cl::Context ctx;
	cl::Device device;
	oclDeviceDesc devDesc;
	cl::CommandQueue cq;
	bool retrievedPlatforms;

private:
	bool retrievePlatforms();
	bool fileContents(const char* progFile, std::string& ret);
	oclDeviceDesc populateDescription(cl::Device dev);
};

template<typename AtomType>
void oclContext::zeroAtomicCounter(oclBufferHolder& counter,
	bool async)
{
	static AtomType d_init = 0;
	cl_int err = cq.enqueueWriteBuffer(counter.buf,
		async ? CL_TRUE : CL_FALSE,
		0, 	sizeof(AtomType), &d_init, nullptr, &counter.evt);

	if(!async)
		counter.evt.wait();

	oclError("Error while zeroing atomic counter value!", err);
}

template<typename AtomType>
void oclContext::readAtomicCounter(const oclBufferHolder& counter,
	AtomType* value, bool async)
{	
	cl_int err = cq.enqueueReadBuffer(counter.buf,
		async ? CL_TRUE : CL_FALSE,
		0, sizeof(AtomType), value, nullptr,
		const_cast<cl::Event*>(&counter.evt));

	if(!async)
		counter.evt.wait();

	oclError("Error while reading atomic counter value!", err);
}