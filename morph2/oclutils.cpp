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