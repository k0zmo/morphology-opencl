#include "oclutils.h"

qreal oclUtils::eventDuration(const QCLEvent& evt)
{
	qreal runDuration = (evt.finishTime() - evt.runTime()) / 1000000.0f;
	return runDuration;
}

QCLImageFormat oclUtils::morphImageFormat()
{
	QCLImageFormat imageFormat
		(QCLImageFormat::Order_R, QCLImageFormat::Type_Normalized_UInt8);
	return imageFormat;
}

int oclUtils::roundUp(int value, int multiple)
{
	int v = value % multiple;
	if (v)
		return value + (multiple - v);
	return value;
}

QCLBuffer oclUtils::setStructuringElement(QCLContext* ctx, const cv::Mat& selement,
	bool shiftCoords, int& seRadiusX, int& seRadiusY)
{
	std::vector<cl_int2> coords;

	seRadiusX = (selement.cols - 1) / 2;
	seRadiusY = (selement.rows - 1) / 2;

	// Przetworz wstepnie element strukturalny
	for(int y = 0; y < selement.rows; ++y)
	{
		const uchar* krow = selement.data + selement.step*y;

		for(int x = 0; x < selement.cols; ++x)
		{
			if(krow[x] == 0)
				continue;

			cl_int2 c = {{x, y}};

			// Dla implementacji z wykorzystaniem obrazow musimy przesunac uklad wspolrzednych
			// elementu strukturalnego
			if(shiftCoords)
			{
				c.s[0] -= seRadiusX;
				c.s[1] -= seRadiusY;
			}

			coords.push_back(c);
		}
	}

	int csize = static_cast<int>(coords.size());
	printf("Structuring element size (number of 'white' pixels): %d (%dx%d) - %lu B\n",
		csize, 2*seRadiusX+1, 2*seRadiusY+1, sizeof(cl_int2) * csize);

	size_t bmuSize = sizeof(cl_int2) * csize;
	cl_ulong limit = ctx->defaultDevice().maximumConstantBufferSize();

	QCLBuffer structuringElement;

	if(bmuSize > limit)
	{
		printf("Structuring element is too big:"
			"%lu B out of available %lu B.", bmuSize, limit);
		//static char tmpBuf[256];
		//snprintf(tmpBuf, sizeof(tmpBuf), "Structuring element is too big:"
		//	"%lu B out of available %lu B.", bmuSize, limit);
		// TODO:
		//oclContext::oclError(tmpBuf, CL_OUT_OF_RESOURCES);
	}
	else
	{
		structuringElement = ctx->createBufferDevice
			(bmuSize, QCLMemoryObject::ReadOnly);
		QCLEvent evt = structuringElement.writeAsync(0, coords.data(), bmuSize);
		evt.waitForFinished();

		printf("Transfering structuring element to device took %.5lf ms\n",
			(evt.finishTime() - evt.runTime()) / 1000000.0f);
	}

	return structuringElement;
}

qreal oclUtils::readAtomicCounter(QCLBuffer& buf, cl_uint& dst)
{
	QCLEvent evt = buf.readAsync(0, &dst, sizeof(cl_uint));
	evt.waitForFinished();
	return oclUtils::eventDuration(evt);
}

qreal oclUtils::zeroAtomicCounter(QCLBuffer& buf)
{
	static int init = 0;
	QCLEvent evt = buf.writeAsync(0, &init, sizeof(cl_uint));
	evt.waitForFinished();
	return oclUtils::eventDuration(evt);
}

qreal oclUtils::copyImage2D(const QCLImage2D& src, QCLImage2D& dst)
{
	QCLEvent evt = const_cast<QCLImage2D&>(src).copyToAsync
		(QRect(0, 0, src.width(), src.height()),
		dst, QPoint(0, 0));
	evt.waitForFinished();
	return oclUtils::eventDuration(evt);
}

qreal oclUtils::copyBuffer(const QCLBuffer& src, QCLBuffer& dst)
{
	QCLEvent evt = const_cast<QCLBuffer&>(src).copyToAsync
		(0, src.size(), dst, 0);
	evt.waitForFinished();
	return oclUtils::eventDuration(evt);
}