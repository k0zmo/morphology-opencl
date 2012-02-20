#pragma once

#include "morphocl.h"

class MorphOpenCLBuffer : public MorphOpenCL
{
public:
	MorphOpenCLBuffer();
	virtual ~MorphOpenCLBuffer();

	/*override*/ virtual bool initOpenCL();
	/*override*/ virtual void setSourceImage(const cv::Mat* src);
	/*override*/ virtual void setSourceImage(const cv::Mat* src, GLuint glresource);
	/*override*/ virtual double morphology(EOperationType opType, cv::Mat& dst, int& iters);

private:
	// Reprezentuje obraz zrodlowy, wszystkie operacje beda odwolywac sie do jego rozmiaru
	struct SBuffer
	{
		const cv::Mat* cpu;
		cl::Buffer gpu;
		int gpuWidth, gpuHeight;
	};
	SBuffer sourceBuffer;	

	//
	// Parametry odczytane z pliku konfiguracyjnego
	//

	// Czy uzyc typu uint zamiast uchar
	bool useUint;
	// Czy uzyc subtract4 zamiast subtract
	bool sub4;

	cl::BufferGL shared;
	GLuint pboStaging;
	GLuint glTexture;

private:
	// Odczyt ze wskazanego bufora do podanej macierzy
	cl_ulong readBack(const cl::Buffer& source, cv::Mat &dst, 
		int dstSizeX, int dstSizeY);

	inline size_t bufferSize() const
	{
		return sourceBuffer.gpuWidth * sourceBuffer.gpuHeight * 
			((useUint) ? sizeof(cl_uint) : sizeof(cl_uchar));
	}

	cl::Buffer createBuffer(cl_mem_flags memFlags);
	cl_ulong copyBuffer(const cl::Buffer& src, cl::Buffer& dst);

	cl_ulong morphologyErode(cl::Buffer& src, cl::Buffer& dst);
	cl_ulong morphologyDilate(cl::Buffer& src, cl::Buffer& dst);
	cl_ulong morphologyOpen(cl::Buffer& src, cl::Buffer& dst);
	cl_ulong morphologyClose(cl::Buffer& src, cl::Buffer& dst);
	cl_ulong morphologyGradient(cl::Buffer& src, cl::Buffer& dst);
	cl_ulong morphologyTopHat(cl::Buffer& src, cl::Buffer& dst);
	cl_ulong morphologyBlackHat(cl::Buffer& src, cl::Buffer& dst);
	cl_ulong morphologyOutline(cl::Buffer& src, cl::Buffer& dst);
	cl_ulong morphologySkeleton(cl::Buffer& src, cl::Buffer& dst, int& iters);
	cl_ulong morphologySkeletonZhangSuen(cl::Buffer& src, cl::Buffer& dst, int& iters);

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
