#pragma once

#include <QCLEvent>
#include <QCLImage>

class oclUtils
{
public:
	static qreal eventDuration(const QCLEvent& evt);
	static QCLImageFormat morphImageFormat();
};