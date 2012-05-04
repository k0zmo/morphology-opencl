#include "configuration.h"

#include <QSettings>


void Configuration::loadConfiguration(const QString& filename)
{
	QSettings s(filename, QSettings::IniFormat);

	// Sekcja [gui]
	maxImageWidth = s.value("gui/maximagewidth").toInt();
	maxImageHeight = s.value("gui/maximageheight").toInt();
	defaultImage = s.value("gui/defaultimage").toString();

	// Sekcja [opencl]
	atomicCounters = s.value("opencl/atomiccounters").toBool();
	glInterop = s.value("opencl/glinterop").toBool();
	workgroupSizeX = s.value("opencl/workgroupsizex").toInt();
	workgroupSizeY = s.value("opencl/workgroupsizey").toInt();

	// Sekcja [kernels-2d]
	erode_2d = s.value("kernels-2d/erode").toString();
	dilate_2d = s.value("kernels-2d/dilate").toString();
	gradient_2d = s.value("kernels-2d/gradient").toString();

	// Sekcja [kernels-1d]
	erode_1d = s.value("kernels-1d/erode").toString();
	dilate_1d = s.value("kernels-1d/dilate").toString();
	gradient_1d = s.value("kernels-1d/gradient").toString();
	subtract_1d = s.value("kernels-1d/subtract").toString();
	hitmiss_1d = s.value("kernels-1d/hitmiss").toString();
	dataType = s.value("kernels-1d/datatype").toInt();
}

void Configuration::saveConfiguration(const QString& filename)
{
	QSettings s(filename, QSettings::IniFormat);

	// Sekcja [gui]
	s.setValue("gui/maximagewidth", maxImageWidth);
	s.setValue("gui/maximageheight", maxImageHeight);
	s.setValue("gui/defaultimage", defaultImage);

	// Sekcja [opencl]
	s.setValue("opencl/atomiccounters", atomicCounters);
	s.setValue("opencl/glinterop", glInterop);
	s.setValue("opencl/workgroupsizex", workgroupSizeX);
	s.setValue("opencl/workgroupsizey", workgroupSizeY);

	// Sekcja [kernels-2d]
	s.setValue("kernels-2d/erode", erode_2d);
	s.setValue("kernels-2d/dilate", dilate_2d);
	s.setValue("kernels-2d/gradient", gradient_2d);

	// Sekcja [kernels-1d]
	s.setValue("kernels-1d/erode", erode_1d);
	s.setValue("kernels-1d/dilate", dilate_1d);
	s.setValue("kernels-1d/gradient", gradient_1d);
	s.setValue("kernels-1d/subtract", subtract_1d);
	s.setValue("kernels-1d/hitmiss", hitmiss_1d);
	s.setValue("kernels-1d/datatype", dataType);
}
