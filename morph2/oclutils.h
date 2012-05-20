#pragma once

#include <qclevent.h>
#include <qclimage.h>
#include <qclbuffer.h>
#include <qclcontext.h>

class oclUtils
{
public:
	static qreal eventDuration(const QCLEvent& evt);
	static QCLImageFormat morphImageFormat();

	static int roundUp(int value, int multiple);

	static QCLBuffer setStructuringElement(QCLContext* ctx, const cv::Mat& selement,
		bool shiftCoords, int& seRadiusX, int& seRadiusY);

	static qreal readAtomicCounter(QCLBuffer& buf, cl_uint& dst);
	static qreal zeroAtomicCounter(QCLBuffer& buf);

	static qreal copyImage2D(const QCLImage2D& src, QCLImage2D& dst);
	static qreal copyBuffer(const QCLBuffer& src, QCLBuffer& dst);
};
