#pragma once

#include <functional>

#include <CL/cl.hpp>
#include <QString>

#include "morphop.h"

class MorphOpenCL
{
public:
	MorphOpenCL();

	std::function<void(const QString&, cl_int)> errorCallback;
	static QString openCLErrorCodeStr(cl_int errcode);

	// Inicjalizuje OpenCL'a
	virtual bool initOpenCL();	

	// Ustawia obraz zrodlowy
	virtual void setSourceImage(const cv::Mat* src) = 0;
	// Ustawia element strukturalny
	int setStructureElement(const cv::Mat& selement);

	void recompile(EOperationType opType, int coordsSize);

	// Wykonanie operacji morfologicznej, zwraca czas trwania
	virtual double morphology(EOperationType opType, cv::Mat& dst, int& iters) = 0;

	// Zczytuje wynik z urzadzenia
	virtual double readBack(cv::Mat &dst, int dstSizeX,
		int dstSizeY, cl_ulong elapsed) = 0;

protected:
	cl::Context context;
	cl::Device dev;
	cl::CommandQueue cq;

	// Bufor ze wspolrzednymi elementu strukturalnego
	cl::Buffer clSeCoords;
	// Ilosc wspolrzednych (rozmiar elementu strukturalnego)
	int csize;

	const cv::Mat* src;
	int kradiusx, kradiusy;

	struct SKernelParameters
	{
		QString programName;
		QString options;
		QString kernelName;
	};

	SKernelParameters erodeParams;
	SKernelParameters dilateParams;

	cl::Kernel kernelSubtract;

	// Standardowe ('cegielki') operacje morfologiczne
	cl::Kernel kernelErode;
	cl::Kernel kernelDilate;

	// Hit-miss
	cl::Kernel kernelOutline;
	cl::Kernel kernelSkeleton_iter[8];
	cl::Kernel kernelSkeleton_pass[2];

protected:
	// Pomocznicza funkcja do zglaszania bledow OpenCL'a
	void clError(const QString& message, cl_int err);

	// Zwraca czas trwania zdarzenia w nanosekundach 
	cl_ulong elapsedEvent(const cl::Event& evt);

	// Zwraca zbudowany program, opcjonalnie mozna podac liste opcji przy kompilowaniu
	cl::Program createProgram(const char* progFile, 
		const char* options = nullptr);

	cl::Program createProgram(const QString& progFile, 
		const QString& options = "");

	// Zeruje wskazany licznik atomowy
	cl_ulong zeroAtomicCounter(const cl::Buffer& clAtomicCounter);

	// Odczytue wartosc ze wzkazanego licznika atomowego
	cl_ulong readAtomicCounter(cl_uint& v, const cl::Buffer& clAtomicCounter);

	// Tworzy kernel'a z podanego programu
	cl::Kernel createKernel(const cl::Program& prog,
		const char* kernelName);

	cl::Kernel createKernel(const cl::Program& prog, 
		const QString& kernelName);
};

class MorphOpenCLImage : public MorphOpenCL
{
public:
	MorphOpenCLImage()
		: MorphOpenCL()
	{ }

	/*override*/ virtual bool initOpenCL();
	/*override*/ virtual void setSourceImage(const cv::Mat* src);
	/*override*/ virtual double morphology(EOperationType opType, cv::Mat& dst, int& iters);
	/*override*/ virtual double readBack(cv::Mat &dst, int dstSizeX,
		int dstSizeY, cl_ulong elapsed);

private:
	// Obraz wejsciowy
	cl::Image2D clSrcImage;
	// Obraz wyjsciowy
	cl::Image2D clDstImage;
	// Tymczasowe obrazy (ping-pong)
	cl::Image2D clTmpImage;
	cl::Image2D clTmp2Image;

	cl::ImageFormat imageFormat;

private:
	// Pomocnicza funkcja do odpalania kerneli do podst. operacji morfologicznych
	cl_ulong executeMorphologyKernel(cl::Kernel* kernel, 
		const cl::Image2D& clSrcImage, cl::Image2D& clDstImage);

	// Pomocnicza funkcja do odpalania kerneli do operacji typu Hit-Miss
	cl_ulong executeHitMissKernel(cl::Kernel* kernel, 
		const cl::Image2D& clSrcImage, cl::Image2D& clDstImage,
		const cl::Buffer* clLut = nullptr,
		cl::Buffer* clAtomicCounter = nullptr);

	// Pomocnicza funkcja do odpalania kernela do odejmowania dwoch obrazow od siebie
	cl_ulong executeSubtractKernel(const cl::Image2D& clAImage, 
		const cl::Image2D& clBImage, cl::Image2D& clDstImage);
};

class MorphOpenCLBuffer : public MorphOpenCL
{
public:
	MorphOpenCLBuffer();

	/*override*/ virtual bool initOpenCL();
	/*override*/ virtual void setSourceImage(const cv::Mat* src);
	/*override*/ virtual double morphology(EOperationType opType, cv::Mat& dst, int& iters);
	/*override*/ virtual double readBack(cv::Mat &dst, int dstSizeX,
		int dstSizeY, cl_ulong elapsed);	

private:
	// Bufor z danymi wejsciowymi
	cl::Buffer clSrc;
	// Bufor z danymi wyjsciowymi
	cl::Buffer clDst;

	// Tymczasowe bufory (ping-pong)
	cl::Buffer clTmp;
	cl::Buffer clTmp2;

	int deviceWidth, deviceHeight;

	int workGroupSizeX;
	int workGroupSizeY;

	enum EReadingMethod
	{
		RM_NotOptimized,
		RM_ReadAligned
	};
	EReadingMethod readingMethod;

	bool local;
	bool useUint;
	bool sub4;

private:
	// Pomocnicza funkcja do odpalania kerneli do podst. operacji morfologicznych
	cl_ulong executeMorphologyKernel(cl::Kernel* kernel, 
		const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer);

	// Pomocnicza funkcja do odpalania kerneli do operacji typu Hit-Miss
	cl_ulong executeHitMissKernel(cl::Kernel* kernel, 
		const cl::Buffer& clSrcBuffer, cl::Buffer& clDstBuffer,
		const cl::Buffer* clLut = nullptr,
		cl::Buffer* clAtomicCounter = nullptr);

	// Pomocnicza funkcja do odpalania kernela do odejmowania dwoch obrazow od siebie
	cl_ulong executeSubtractKernel(const cl::Buffer& clABuffer,
		const cl::Buffer& clBBuffer, cl::Buffer& clDstBuffer);
};