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
	// Konstruktor domyslny - brak jakichkolwiek alokacji
	oclContext();

	// Zwraca liste dostepnych platform
	bool retrievePlatforms(std::vector<oclPlatformDesc>& out);

	// Zwraca liste dostepnych urzadzen dla danej platformy
	void retrieveDevices(size_t platformId, std::vector<oclDeviceDesc>& out);

	// Tworzy kontekst w domyslnej platformie
	bool createContext();
	// Tworzy kontekst w podanej platformie
	bool createContext(size_t platformId);
	// Tworzy kontekst w podanej platformie, ktory dzieli zasoby wraz z aktualnie
	// aktywnym kontekstem OpenGL'a
	bool createContextGL(size_t platformId);

	// Wybiera urzadzenie obliczeniowe
	void chooseDevice(size_t deviceId);
	// Tworzy kolejke - wymagane jest wczesniejsze wybranie urzadzenia obliczeniowego
	bool createCommandQueue(bool profiling);

	// Tworzy program (lub zwraca uprzednio zcache'owny) o podanej nazwie i opcjach budowania
	// Mozna wymusic jego ponowna rekompilacje
	cl::Program createProgram(const char* progFile,
		const char* options, bool forceBuild = false);
	// Zwraca obiekt kernela z podanego programu
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

	// Tworzy pusty obraz na urzadzeniu obliczeniowym
	oclImage2DHolder createDeviceImage(int width, int height,
		oclMemoryAccess access);

	// Tworzy obraz z podanej tesktury OpenGL'a
	oclImage2DHolder createDeviceImageGL(GLuint resource,
		oclMemoryAccess access);

	// Tworzy pusty obiekt bufora na urzadzeniu obliczeniowym
	oclBufferHolder createDeviceBuffer(int size, oclMemoryAccess access);

	// Kopiuje zawartosc jednego obrazu do drugiego bez udzialu hosta
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
	// Lista pobranych platform
	std::vector<cl::Platform> pls;
	// Lista zcache'owanych programow
	std::map<std::string, std::map<std::string, cl::Program>> programs;
	// Utworzony kontekst
	cl::Context ctx;
	// Wybrane urzadzenie
	cl::Device device;
	// Opis wybranego urzadzenia
	oclDeviceDesc devDesc;
	// Jego kolejka
	cl::CommandQueue cq;
	// Czy pobrano liste dostepnych platform
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
