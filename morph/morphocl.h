#pragma once

#include <CL/cl.hpp>
#include <QString>

#include "morphop.h"

class MorphOpenCL
{
public:
	MorphOpenCL()
		: src(nullptr), 
		kradiusx(0),
		kradiusy(0)
	{ }

	// Inicjalizuje OpenCL'a
	bool initOpenCL();
	// Ustawia obraz zrodlowy
	void setSourceImage(const cv::Mat* src);
	// Ustawia element strukturalny
	void setStructureElement(const cv::Mat& selement);
	// Wykonanie operacji morfologicznej, zwraca czas trwania
	double morphology(EOperationType opType, cv::Mat& dst, int& iters);

private:
	cl::Context context;
	cl::Device dev;
	cl::CommandQueue cq;

	cl::Kernel kernelSubtract;
	cl::Kernel kernelAddHalf;

	// Standardowe ('cegielki') operacje morfologiczne
	cl::Kernel kernelErode;
	cl::Kernel kernelDilate;
	// Hit-miss
	cl::Kernel kernelRemove;
	cl::Kernel kernelSkeleton_iter[8];

	// Bufor z danymi wejsciowymi
	cl::Buffer clSrc;
	// Bufor z danymi wyjsciowymi
	cl::Buffer clDst;
	// Bufor ze wspolrzednymi elementu strukturalnego
	cl::Buffer clSeCoords;
	// Ilosc wspolrzednych (rozmiar elementu strukturalnego)
	size_t csize;

	// Tymczasowe bufory (ping-pong)
	cl::Buffer clTmp;
	cl::Buffer clTmp2;

	const cv::Mat* src;
	int kradiusx, kradiusy;

private:
	// Pomocznicza funkcja do zglaszania bledow OpenCL'a
	void clError(const QString& message, cl_int err);

	// Pomocnicza funkcja do odpalania kerneli do podst. operacji morfologicznych
	cl_ulong executeMorphologyKernel(cl::Kernel* kernel, 
		const cl::Buffer& clBufferSrc, cl::Buffer& clBufferDst);

	// Pomocnicza funkcja do odpalania kerneli do operacji typu Hit-Miss
	cl_ulong executeHitMissKernel(cl::Kernel* kernel, const cl::Buffer& clBufferSrc,
		cl::Buffer& clBufferDst);

	// Pomocnicza funkcja do odpalania kernela do odejmowania dwoch obrazow od siebie
	cl_ulong executeSubtractKernel(const cl::Buffer& clBufferA, const cl::Buffer& clBufferB,
		cl::Buffer& clBufferDst);

	// Pomocnicza funkcja do odpalania kernela do nalozenia na siebie dwoch obrazow (do prezentacji szkieletyzacji)
	cl_ulong executeAddHalfKernel(const cl::Buffer& clBufferSrc,
		cl::Buffer& clBufferDst);

	// Zwraca czas trwania zdarzenia w nanosekundach 
	cl_ulong elapsedEvent(const cl::Event& evt);
};