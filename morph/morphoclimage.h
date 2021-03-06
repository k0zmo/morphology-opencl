#pragma once

#include "morphocl.h"

class MorphOpenCLImage : public MorphOpenCL
{
public:
	MorphOpenCLImage();
	virtual ~MorphOpenCLImage();

	virtual bool initOpenCL();
	virtual void setSourceImage(const cv::Mat* src);
	virtual void setSourceImage(const cv::Mat* src, GLuint glresource);
	virtual double morphology(Morphology::EOperationType opType,
		cv::Mat& dst, int& iters);

private:
	// Reprezentuje obraz zrodlowy, wszystkie operacje beda odwolywac sie do jego rozmiaru czy formatu
	struct SImage
	{
		const cv::Mat* cpu;
		cl::ImageFormat format;
		cl::Image2D gpu;
	};
	SImage sourceImage;	

	cl::Image2DGL shared;

private:
	// Odczyt ze wskazanej tekstury do podanej macierzy
	cl_ulong readBack(cl::Image2D& source, cv::Mat &dst);

	cl_ulong copyImage2D(const cl::Image2D& src, cl::Image2D& dst);
	cl::Image2D createImage2D(cl_mem_flags memFlags);

	cl_ulong morphologyErode(cl::Image2D& src, cl::Image2D& dst);
	cl_ulong morphologyDilate(cl::Image2D& src, cl::Image2D& dst);
	cl_ulong morphologyOpen(cl::Image2D& src, cl::Image2D& dst);
	cl_ulong morphologyClose(cl::Image2D& src, cl::Image2D& dst);
	cl_ulong morphologyGradient(cl::Image2D& src, cl::Image2D& dst);
	cl_ulong morphologyTopHat(cl::Image2D& src, cl::Image2D& dst);
	cl_ulong morphologyBlackHat(cl::Image2D& src, cl::Image2D& dst);
	cl_ulong morphologyOutline(cl::Image2D& src, cl::Image2D& dst);
	cl_ulong morphologySkeleton(cl::Image2D& src, cl::Image2D& dst, int& iters);
	cl_ulong morphologySkeletonZhangSuen(cl::Image2D& src, cl::Image2D& dst, int& iters);

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

	cl_ulong executeBayerFilter(cl::Kernel* kernel, 
		const cl::Image2D& clSrcImage, const cl::Image2D& clDstImage);
};