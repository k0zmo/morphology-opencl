#pragma once

#include <qclevent.h>
#include <qclimage.h>

class oclUtils
{
public:
	static qreal eventDuration(const QCLEvent& evt);
	static QCLImageFormat morphImageFormat();
};
