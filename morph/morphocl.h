#pragma once

#include <functional>

#include <CL/cl.hpp>
#include <QString>

#include "morphoperators.h"

class MorphOpenCL
{
public:
	MorphOpenCL();
	virtual ~MorphOpenCL();

	// Callback wywolywany gdy wystapi blad zgloszony przez OpenCLa
	typedef std::function<void(const QString&, cl_int)> ErrorCallback;

	void setErrorCallback(ErrorCallback cb)
	{ this->cb = cb; }

	static QString openCLErrorCodeStr(cl_int errcode);

	// Inicjalizuje OpenCL'a
	virtual bool initOpenCL();	

	// Ustawia obraz zrodlowy do operacji morfologicznych
	virtual void setSourceImage(const cv::Mat* src) = 0;
	virtual void setSourceImage(const cv::Mat* src, GLuint glresource) = 0;

	// Ustawia czy przed filtracja wlasciwa wykonana zostanie interpolacja bayera
	void setBayerFilter(Morphology::EBayerCode code)
	{ bayerFilter = code; }

	// Ustawia element strukturalny
	int setStructuringElement(const cv::Mat& selement);

	// Wykonanie operacji morfologicznej, zwraca czas trwania
	virtual double morphology(Morphology::EOperationType opType, cv::Mat& dst, int& iters) = 0;

	// Rekompiluje kod kernela odpowiedzialny za wskazana operacje
	// (Ma sens dla dylatacji i erozji)
	void recompile(Morphology::EOperationType opType, int coordsSize);

	bool usingShared() const { return useShared; }

	inline void setWorkGroupSize(int x, int y)
	{ workGroupSizeX = x; workGroupSizeY = y; }

	bool error;

protected:
	cl::Context context;
	cl::Device dev;
	cl::CommandQueue cq;
	ErrorCallback cb;

	// Bufor ze wspolrzednymi elementu strukturalnego
	cl::Buffer clStructuringElementCoords;
	// Ilosc wspolrzednych (rozmiar elementu strukturalnego)
	int csize;

	int kradiusx, kradiusy;
	
	// Rozmiar grupy roboczej
	int workGroupSizeX;
	int workGroupSizeY;

	int sharedw, sharedh;
	bool useShared;
	Morphology::EBayerCode bayerFilter;

	struct SKernelParameters
	{
		QString programName;
		QString options;
		QString kernelName;
		bool needRecompile;
	};

	SKernelParameters erodeParams;
	SKernelParameters dilateParams;
	SKernelParameters gradientParams;

	cl::Kernel kernelSubtract;
	cl::Kernel kernelBayer[4];

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
