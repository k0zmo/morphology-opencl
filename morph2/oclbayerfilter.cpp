#include "oclbayerfilter.h"

QCLImageFormat morphImageFormat()
{
	QCLImageFormat imageFormat
			(QCLImageFormat::Order_R, QCLImageFormat::Type_Normalized_UInt8);
	return imageFormat;
}

//oclFilter::oclFilter(QCLContext* ctx)
//	: d_ctx(ctx)
//	, d_localSize(8, 8)
//	, d_src(nullptr)
//	//, roi(cvu::WholeImage)
//	//, ownsOutput(true)
//{
//}

//oclFilter::~oclFilter()
//{
//}

//void oclFilter::setSourceImage(
//	const QCLImage2D& src)
//{
//	d_src = &src;

////	if(roi == cvu::WholeImage)
////	{
////		this->roi.width = src.width;
////		this->roi.height = src.height;
////	}
////	else
////	{
////		// TODO
////		//	roi.x = std::min(roi.x, src.width);
////		//	roi.y = std::min(roi.y, src.height);
////		//	roi.width = [something]
////	}
//}

////void oclFilter::setOutputDeviceImage(const oclImage2DHolder& img)
////{
////	dst = img;
////	ownsOutput = false;
////}

//QCLWorkSize oclFilter::computeOffset(
//	int minBorderX, int minBorderY)
//{
////	// TODO jesli roi jest mniejszy niz obraz nie trzeba go przesuwac

////	cv::Rect r(roi);
////	r.x = std::max(minBorderX, r.x);
////	r.y = std::max(minBorderY, r.y);

////	return cl::NDRange(r.x, r.y);

//	return QCLWorkSize(minBorderX, minBorderY);
//}

//QCLWorkSize oclFilter::computeGlobal(
//	int minBorderX, int minBorderY)
//{
////	// TODO jesli roi jest inny niz WholeImage

////	cl::NDRange local = ctx->workgroupSize();

////	int gx = oclContext::roundUp(roi.width - 2*minBorderX, local[0]);
////	int gy = oclContext::roundUp(roi.height - 2*minBorderY, local[1]);

////	return cl::NDRange(gx, gy);

//	QCLWorkSize global(d_dst.width() - 2*minBorderX,
//					   d_dst.height() - 2*minBorderY);
//	global = global.roundTo(d_localSize);
//	return global;
//}

//void oclFilter::prepareDestinationHolder()
//{
//	//if(ownsOutput)
//	{
//		//dst = ctx->createDeviceImage(
//		//	src->width, src->height, ReadWrite);

//		QSize dstSize(d_src->width(), d_src->height());
//		d_dst = d_ctx->createImage2DDevice
//				(morphImageFormat(), dstSize, QCLMemoryObject::ReadWrite);
//	}
////	else
////	{
////		ctx->acquireGLTexture(dst);
////	}
//}

//void oclFilter::finishUpDestinationHolder()
//{
//	// Musimy zwolnic zasob dla OpenGLa
////	if(!ownsOutput)
////		ctx->releaseGLTexture(dst);
//}






//oclBayerFilter::oclBayerFilter(
//	QCLContext* ctx)
//	: oclFilter(ctx)
//	, d_kernel(nullptr)
//{
//	QCLProgram program = ctx->createProgramFromSourceFile("kernels/2d/bayer.cl");
//	if(program.isNull() ||
//	   program.build(QList<QCLDevice>(), "-Ikernels/2d/"))
//	{
//		if(ctx->lastError() != CL_SUCCESS)
//			return;

//		// I wyciagnij z niego kernele
//		d_kernels[cvu::BC_RedGreen  - 1] = program.createKernel
//			("convert_rg2gray");
//		d_kernels[cvu::BC_GreenRed  - 1] = program.createKernel
//			("convert_gr2gray");
//		d_kernels[cvu::BC_BlueGreen - 1] = program.createKernel
//			("convert_bg2gray");
//		d_kernels[cvu::BC_GreenBlue - 1] = program.createKernel
//			("convert_gb2gray");
//	}
//}

//QCLEvent oclBayerFilter::run()
//{
//	if(!d_src)
//		return QCLEvent();

//	if(!d_kernel)
//	{
//		// Passthrough
//		//if(!ownsOutput)
//		//{
//		//	// TODO
//		//}
//		//else
//		{
//			d_dst = *d_src;
//		}
//		return QCLEvent();
//	}

////	prepareDestinationHolder();

//	QSize dstSize(d_src->width(), d_src->height());
//	d_dst = d_ctx->createImage2DDevice
//			(morphImageFormat(), dstSize, QCLMemoryObject::ReadWrite);

//	QCLWorkSize offset(computeOffset(1, 1));
//	QCLWorkSize global(computeGlobal(1, 1));

//	d_kernel->setLocalWorkSize(localWorkSize());
//	d_kernel->setGlobalWorkSize(global);
//	d_kernel->setGlobalWorkOffset(offset);

//	d_kernel->setArg(0, *d_src);
//	d_kernel->setArg(1, d_dst);
//	QCLEvent evt = d_kernel->run();

//	evt.waitForFinished();
//	return evt;
//}

//void oclBayerFilter::setBayerFilter(
//	cvu::EBayerCode bc)
//{
//	if(bc == cvu::BC_None)
//		d_kernel = nullptr;
//	else
//		d_kernel = &d_kernels[bc - 1];
//}
