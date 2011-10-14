#pragma once

#include <functional>
#include <CL/cl.hpp>
#include <QString>
#include <unordered_map>

#include "morphop.h"

class MorphOpenCL
{
public:
	MorphOpenCL()
		: src(nullptr), 
		kradiusx(0),
		kradiusy(0),
		errorCallback(nullptr)
	{ }

	std::function<void(const QString&, cl_int)> errorCallback;

	// Inicjalizuje OpenCL'a
	virtual bool initOpenCL(cl_device_type dt);
	// Ustawia obraz zrodlowy
	virtual void setSourceImage(const cv::Mat* src) = 0;
	// Ustawia element strukturalny
	int setStructureElement(const cv::Mat& selement);

	// Wykonanie operacji morfologicznej, zwraca czas trwania
	virtual double morphology(EOperationType opType, cv::Mat& dst, int& iters) = 0;

protected:
	cl::Context context;
	cl::Device dev;
	cl::CommandQueue cq;

	// Hash-map'a zbudowanych programow 
	// (kluczem jest sciezka do pliku, z ktorego program zbudowano)
	std::unordered_map<std::string, cl::Program> programs;

	// Bufor ze wspolrzednymi elementu strukturalnego
	cl::Buffer clSeCoords;
	// Ilosc wspolrzednych (rozmiar elementu strukturalnego)
	size_t csize;

	const cv::Mat* src;
	int kradiusx, kradiusy;

protected:
	// Pomocznicza funkcja do zglaszania bledow OpenCL'a
	void clError(const QString& message, cl_int err);

	// Zwraca czas trwania zdarzenia w nanosekundach 
	cl_ulong elapsedEvent(const cl::Event& evt);

	// Zwraca zbudowany program, opcjonalnie mozna podac liste opcji przy kompilowaniu
	cl::Program createProgram(const char* progFile, 
		const char* options = nullptr);

	// Zeruje wskazany licznik atomowy
	cl_ulong zeroAtomicCounter(const cl::Buffer& clAtomicCounter);

	// Odczytue wartosc ze wzkazanego licznika atomowego
	cl_ulong readAtomicCounter(cl_uint& v, const cl::Buffer& clAtomicCounter);

	// Tworzy kernel'a z podanego programu
	cl::Kernel createKernel(const cl::Program& prog,
		const char* kernelName);
};

class MorphOpenCLImage : public MorphOpenCL
{
public:
	MorphOpenCLImage()
		: MorphOpenCL()
	{ }

	/*override*/ virtual bool initOpenCL(cl_device_type dt);
	/*override*/ virtual void setSourceImage(const cv::Mat* src);
	/*override*/ virtual double morphology(EOperationType opType, cv::Mat& dst, int& iters);

private:
	cl::Kernel kernelSubtract;
	cl::Kernel kernelDiffPixels;

	// Standardowe ('cegielki') operacje morfologiczne
	cl::Kernel kernelErode;
	cl::Kernel kernelDilate;

	// Hit-miss
	cl::Kernel kernelThinning;
	cl::Kernel kernelSkeleton_iter[8];

	// Obraz wejsciowy
	cl::Image2D clSrcImage;
	// Obraz wyjsciowy
	cl::Image2D clDstImage;
	// Tymczasowe obrazy (ping-pong)
	cl::Image2D clTmpImage;
	cl::Image2D clTmp2Image;

private:
	// Pomocnicza funkcja do odpalania kerneli do podst. operacji morfologicznych
	cl_ulong executeMorphologyKernel(cl::Kernel* kernel, 
		const cl::Image2D& clSrcImage, cl::Image2D& clDstImage);

	// Pomocnicza funkcja do odpalania kerneli do operacji typu Hit-Miss
	cl_ulong executeHitMissKernel(cl::Kernel* kernel, 
		const cl::Image2D& clSrcImage, cl::Image2D& clDstImage);

	// Pomocnicza funkcja do odpalania kernela do odejmowania dwoch obrazow od siebie
	cl_ulong executeSubtractKernel(const cl::Image2D& clAImage, 
		const cl::Image2D& clBImage, cl::Image2D& clDstImage);

	cl_ulong executeDiffPixelsKernel(const cl::Image2D& clAImage,
		const cl::Image2D& clBImage, const cl::Buffer& clAtomicCounter);
};

class MorphOpenCLBuffer : public MorphOpenCL
{
public:
	MorphOpenCLBuffer()
		: MorphOpenCL()
	{ }

	/*override*/ virtual bool initOpenCL(cl_device_type dt);
	/*override*/ virtual void setSourceImage(const cv::Mat* src);
	/*override*/ virtual double morphology(EOperationType opType, cv::Mat& dst, int& iters);

private:
	cl::Kernel kernelSubtract;
	cl::Kernel kernelDiffPixels;

	// Standardowe ('cegielki') operacje morfologiczne
	cl::Kernel kernelErode;
	cl::Kernel kernelDilate;

	// Hit-miss
	cl::Kernel kernelThinning;
	cl::Kernel kernelSkeleton_iter[8];

	// Bufor z danymi wejsciowymi
	cl::Buffer clSrc;
	// Bufor z danymi wyjsciowymi
	cl::Buffer clDst;

	// Tymczasowe bufory (ping-pong)
	cl::Buffer clTmp;
	cl::Buffer clTmp2;	

private:
	// Pomocnicza funkcja do odpalania kerneli do podst. operacji morfologicznych
	cl_ulong executeMorphologyKernel(cl::Kernel* kernel, 
		const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer);

	// Pomocnicza funkcja do odpalania kerneli do operacji typu Hit-Miss
	cl_ulong executeHitMissKernel(cl::Kernel* kernel, 
		const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer);

	// Pomocnicza funkcja do odpalania kernela do odejmowania dwoch obrazow od siebie
	cl_ulong executeSubtractKernel(const cl::Buffer& clABuffer,
		const cl::Buffer& clBBuffer, cl::Buffer& clDstBuffer);

	cl_ulong executeDiffPixelsKernel(const cl::Buffer& clABuffer,
		const cl::Buffer& clBBuffer, const cl::Buffer& clAtomicCounter);
};