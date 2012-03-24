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
	dataType = s.value("opencl/datatype").toInt();

	/// Sekcja [kernel-buffer2D]
	erode_2d = s.value("kernel-buffer2D/erode").toString();
	dilate_2d = s.value("kernel-buffer2D/dilate").toString();
	gradient_2d = s.value("kernel-buffer2D/gradient").toString();

	// Sekcja [kernel-buffer1D]
	erode_1d = s.value("kernel-buffer1D/erode").toString();
	dilate_1d = s.value("kernel-buffer1D/dilate").toString();
	gradient_1d = s.value("kernel-buffer1D/gradient").toString();
	subtract_1d = s.value("kernel-buffer1D/subtract").toString();
	hitmiss_1d = s.value("kernel-buffer1D/hitmiss").toString();
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
	s.setValue("opencl/datatype", dataType);

	/// Sekcja [kernel-buffer2D]
	s.setValue("kernel-buffer2D/erode", erode_2d);
	s.setValue("kernel-buffer2D/dilate", dilate_2d);
	s.setValue("kernel-buffer2D/gradient", gradient_2d);

	// Sekcja [kernel-buffer1D]
	s.setValue("kernel-buffer1D/erode", erode_1d);
	s.setValue("kernel-buffer1D/dilate", dilate_1d);
	s.setValue("kernel-buffer1D/gradient", gradient_1d);
	s.setValue("kernel-buffer1D/subtract", subtract_1d);
	s.setValue("kernel-buffer1D/hitmiss", hitmiss_1d);
}