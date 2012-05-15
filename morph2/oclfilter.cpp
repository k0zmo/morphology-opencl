#include "oclfilter.h"
#include "oclutils.h"

#include <qclcontextgl.h>

oclFilter::oclFilter(QCLContext* ctx)
	: d_ctx(ctx)
	, d_localSize(8, 8)
	, d_src(nullptr)
	//, roi(cvu::WholeImage)
	, ownsOutput(true)
{
}

oclFilter::~oclFilter()
{
}

void oclFilter::setSourceImage(const QCLImage2D& src)
{
	d_src = &src;
//	if(roi == cvu::WholeImage)
//	{
//		this->roi.width = src.width;
//		this->roi.height = src.height;
//	}
//	else
//	{
//		// TODO
//		//	roi.x = std::min(roi.x, src.width);
//		//	roi.y = std::min(roi.y, src.height);
//		//	roi.width = [something]
//	}
}

void oclFilter::setOutputDeviceImage(const QCLImage2D& img)
{
	d_dst = img;
	ownsOutput = false;
}

QCLWorkSize oclFilter::computeOffset(
	int minBorderX, int minBorderY)
{
//	// TODO jesli roi jest mniejszy niz obraz nie trzeba go przesuwac

//	cv::Rect r(roi);
//	r.x = std::max(minBorderX, r.x);
//	r.y = std::max(minBorderY, r.y);

//	return cl::NDRange(r.x, r.y);

	return QCLWorkSize(minBorderX, minBorderY);
}

QCLWorkSize oclFilter::computeGlobal(
	int minBorderX, int minBorderY)
{
//	// TODO jesli roi jest inny niz WholeImage

//	cl::NDRange local = ctx->workgroupSize();

//	int gx = oclContext::roundUp(roi.width - 2*minBorderX, local[0]);
//	int gy = oclContext::roundUp(roi.height - 2*minBorderY, local[1]);

//	return cl::NDRange(gx, gy);

	QCLWorkSize global(d_dst.width()  - 2*minBorderX,
					   d_dst.height() - 2*minBorderY);
	global = global.roundTo(d_localSize);
	return global;
}

void oclFilter::prepareDestinationHolder()
{
	if(ownsOutput)
	{
		QSize dstSize(d_src->width(), d_src->height());
		d_dst = d_ctx->createImage2DDevice
			(oclUtils::morphImageFormat(), dstSize, QCLMemoryObject::ReadWrite);
	}
	else
	{
		QCLEvent acEvt = static_cast<QCLContextGL*>(d_ctx)->acquire(d_dst);
		acEvt.waitForFinished();
	}
}

void oclFilter::finishUpDestinationHolder()
{
	// Musimy zwolnic zasob dla OpenGLa
	if(!ownsOutput)
	{
		QCLEvent reEvt = static_cast<QCLContextGL*>(d_ctx)->release(d_dst);
		reEvt.waitForFinished();
	}
}
