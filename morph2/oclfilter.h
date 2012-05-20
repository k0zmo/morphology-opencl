#pragma once

#include "cvutils.h"

#include <qclcontext.h>
#include <qclevent.h>
#include <QRect>

static const QRect WholeImage(0, 0, -1, -1);

class oclBaseFilter
{
	Q_DISABLE_COPY(oclBaseFilter)
public:
	oclBaseFilter(QCLContext* ctx);
	virtual ~oclBaseFilter();

	virtual qreal run() = 0;

protected:
	QCLWorkSize computeOffset(int minBorderX, int minBorderY);
	QCLWorkSize computeGlobal(int minBorderX, int minBorderY);

protected:
	QCLContext* d_ctx;
	QCLWorkSize d_localSize;
	QRect d_roi;
	QSize d_size;
	bool d_ownsOutput;
};

class oclFilter : public oclBaseFilter
{
	Q_DISABLE_COPY(oclFilter)
public:
	oclFilter(QCLContext* ctx);
	virtual ~oclFilter();

	void setSourceImage(const QCLImage2D& src,
		const QRect& roi = WholeImage);

	void setOutputDeviceImage(const QCLImage2D& img);
	void unsetOutputDeviceImage() { d_ownsOutput = true; }

	QCLImage2D outputDeviceImage() const { return d_dst; }

	QCLWorkSize localWorkSize() const { return d_localSize; }
	void setLocalWorkSize(const QCLWorkSize& local) { d_localSize = local; }

protected:
	void prepareDestinationHolder();
	void finishUpDestinationHolder();

protected:
	const QCLImage2D* d_src;
	QCLImage2D d_dst;
};

class oclFilterBuffer : public oclBaseFilter
{
	Q_DISABLE_COPY(oclFilterBuffer)
public:
	oclFilterBuffer(QCLContext* ctx);
	virtual ~oclFilterBuffer();

	void setSourceImage(const QCLBuffer& src, const QSize& size,
		const QRect& roi = WholeImage);

	void setOutputDeviceImage(const QCLBuffer& img);
	void unsetOutputDeviceImage() { d_ownsOutput = true; }

	QCLBuffer outputDeviceImage() const { return d_dst; }

	QCLWorkSize localWorkSize() const { return d_localSize; }
	void setLocalWorkSize(const QCLWorkSize& local) { d_localSize = local; }

protected:
	void prepareDestinationHolder();
	void finishUpDestinationHolder();

protected:
	const QCLBuffer* d_src;
	QCLBuffer d_dst;
	QSize dstSize;
};