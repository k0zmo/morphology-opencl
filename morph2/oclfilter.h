#pragma once

#include "cvutils.h"

#include <qclcontext.h>
#include <qclevent.h>

class oclFilter
{
	Q_DISABLE_COPY(oclFilter)
public:
	oclFilter(QCLContext* ctx);
	virtual ~oclFilter();

	virtual qreal run() = 0;

	void setSourceImage(const QCLImage2D& src);
		//const QRect& roi = cvu::WholeImage);

	void setOutputDeviceImage(const QCLImage2D& img);
	void unsetOutputDeviceImage() { ownsOutput = true; }

	QCLImage2D outputDeviceImage() const { return d_dst; }

	QCLWorkSize localWorkSize() const { return d_localSize; }
	void setLocalWorkSize(const QCLWorkSize& local) { d_localSize = local; }

protected:
	QCLWorkSize computeOffset(int minBorderX, int minBorderY);
	QCLWorkSize computeGlobal(int minBorderX, int minBorderY);

	void prepareDestinationHolder();
	void finishUpDestinationHolder();

protected:
	QCLContext* d_ctx;
	QCLWorkSize d_localSize;

	const QCLImage2D* d_src;
	QCLImage2D d_dst;

//	//QRect roi;
	bool ownsOutput;
};
