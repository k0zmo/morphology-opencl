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

	// Ustawia obraz zrodlowy do operacji morfologicznych
	virtual void setSourceImage(const cv::Mat* src) = 0;

	// Ustawia element strukturalny
	int setStructuringElement(const cv::Mat& selement);

	// Wykonanie operacji morfologicznej, zwraca czas trwania
	virtual double morphology(EOperationType opType, cv::Mat& dst, int& iters) = 0;

	// Rekompiluje kod kernela odpowiedzialny za wskazana operacje
	// (Ma sens dla dylatacji i erozji)
	void recompile(EOperationType opType, int coordsSize);

	inline void setWorkGroupSize(int x, int y)
	{ workGroupSizeX = x; workGroupSizeY = y; }

protected:
	cl::Context context;
	cl::Device dev;
	cl::CommandQueue cq;

	// Bufor ze wspolrzednymi elementu strukturalnego
	cl::Buffer clStructuringElementCoords;
	// Ilosc wspolrzednych (rozmiar elementu strukturalnego)
	int csize;

	int kradiusx, kradiusy;

	// Rozmiar grupy roboczej
	int workGroupSizeX;
	int workGroupSizeY;

	struct SKernelParameters
	{
		QString programName;
		QString options;
		QString kernelName;
	};

	SKernelParameters erodeParams;
	SKernelParameters dilateParams;
	SKernelParameters gradientParams;

	cl::Kernel kernelSubtract;

	// Standardowe ('cegielki') operacje morfologiczne
	cl::Kernel kernelErode;
	cl::Kernel kernelDilate;
	cl::Kernel kernelGradient;

	// Hit-miss
	cl::Kernel kernelOutline;
	cl::Kernel kernelSkeleton_iter[8];
	cl::Kernel kernelSkeleton_pass[2];
	cl_ulong maxConstantBufferSize;

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

int roundUp(int value, int multiple);
