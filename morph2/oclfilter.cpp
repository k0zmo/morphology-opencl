#include "oclfilter.h"
#include "oclutils.h"

#include <qclcontextgl.h>

oclBaseFilter::oclBaseFilter(QCLContext* ctx)
	: d_ctx(ctx)
	, d_localSize(8, 8)
	, d_roi(WholeImage)
	, d_size(0, 0)
	, d_ownsOutput(true)
{
}

oclBaseFilter::~oclBaseFilter()
{
}

QCLWorkSize oclBaseFilter::computeOffset(
	int minBorderX, int minBorderY)
{
	QRect r(d_roi);

	return QCLWorkSize(
		qMax(minBorderX, r.x()),
		qMax(minBorderY, r.y()));
}

QCLWorkSize oclBaseFilter::computeGlobal(
	int minBorderX, int minBorderY)
{
	int gx = d_roi.width();
	int gy = d_roi.height();

	// if(minBorderX > d_roi.x())
		// gx -= minBorderX;
	// if(minBorderY > d_roi.y())
		// gy -= minBorderY;

	// if(gx >= d_roi.width() + minBorderX)
		// gx -= minBorderX;
	// if(gy >= d_roi.height() + minBorderY)
		// gy -= minBorderY;
		
	int marginLeft = d_roi.x() - minBorderX;
	int marginTop = d_roi.y() - minBorderY;
	int marginRight = d_roi.width() + minBorderX;
	int marginBottom = d_roi.height() + minBorderY;
		
	if(marginLeft < 0)
		gx -= marginLeft;
	if(marginTop < 0)
		gy -= marginTop;
	if(marginRight > d_size.width())
		gx -= marginRight;
	if(marginBottom > d_size.height())
		gy -= marginBottom;

	QCLWorkSize global(gx, gy);
	return global.roundTo(d_localSize);
}

oclFilter::oclFilter(QCLContext* ctx)
	: oclBaseFilter(ctx)
	, d_src(nullptr)
{
}

oclFilter::~oclFilter()
{
}

void oclFilter::setSourceImage(const QCLImage2D& src, const QRect& roi)
{
	d_src = &src;
	d_size = QSize(src.width(), src.height());

	if(roi == WholeImage)
	{
		d_roi.setWidth(d_size.width());
		d_roi.setHeight(d_size.height());
	}
	else
	{
		int imageWidth = d_size.width();
		int imageHeight = d_size.height();

		int rx = qBound(0, roi.x(), imageWidth-1);
		int ry = qBound(0, roi.y(), imageHeight-1);
		int rwidth = qBound(1, roi.width(), imageWidth-rx);
		int rheight = qBound(1, roi.height(), imageHeight-ry);

		d_roi = QRect(rx, ry, rwidth, rheight);
	}
}

void oclFilter::setOutputDeviceImage(const QCLImage2D& img)
{
	d_dst = img;
	d_ownsOutput = false;
}

void oclFilter::prepareDestinationHolder()
{
	if(d_ownsOutput)
	{
		d_dst = d_ctx->createImage2DDevice
			(oclUtils::morphImageFormat(), 
			 d_size, QCLMemoryObject::ReadWrite);
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
	if(!d_ownsOutput)
	{
		QCLEvent reEvt = static_cast<QCLContextGL*>(d_ctx)->release(d_dst);
		reEvt.waitForFinished();
	}
}

// ____________________________________________________________________________

oclFilterBuffer::oclFilterBuffer(QCLContext* ctx)
	: oclBaseFilter(ctx)
	, d_src(nullptr)
{
}

oclFilterBuffer::~oclFilterBuffer()
{
}

void oclFilterBuffer::setSourceImage
	(const QCLBuffer& src, const QSize& size,
	 const QRect& roi)
{
	d_src = &src;
	d_size = size;

	if(roi == WholeImage)
	{
		d_roi.setWidth(d_size.width());
		d_roi.setHeight(d_size.height());
	}
	else
	{
		int imageWidth = d_size.width();
		int imageHeight = d_size.height();

		int rx = qBound(0, roi.x(), imageWidth-1);
		int ry = qBound(0, roi.y(), imageHeight-1);
		int rwidth = qBound(1, roi.width(), imageWidth-rx);
		int rheight = qBound(1, roi.height(), imageHeight-ry);

		d_roi = QRect(rx, ry, rwidth, rheight);
	}
}

void oclFilterBuffer::setOutputDeviceImage(const QCLBuffer& img)
{
	d_dst = img;
	d_ownsOutput = false;
}

void oclFilterBuffer::prepareDestinationHolder()
{
	if(d_ownsOutput)
	{
		size_t bufferSize = d_size.width() * d_size.height();
		d_dst = d_ctx->createBufferDevice
			(bufferSize, QCLMemoryObject::ReadWrite);
	}
	else
	{
		QCLEvent acEvt = static_cast<QCLContextGL*>(d_ctx)->acquire(d_dst);
		acEvt.waitForFinished();
	}
}

void oclFilterBuffer::finishUpDestinationHolder()
{
	// Musimy zwolnic zasob dla OpenGLa
	if(!d_ownsOutput)
	{
		QCLEvent reEvt = static_cast<QCLContextGL*>(d_ctx)->release(d_dst);
		reEvt.waitForFinished();
	}
}