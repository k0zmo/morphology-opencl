#pragma  once

#include <QString>

struct Configuration
{
	// Sekcja [gui]
	int maxImageWidth;
	int maxImageHeight;
	QString defaultImage;

	// Sekcja [opencl]
	bool atomicCounters;
	int workgroupSizeX;
	int workgroupSizeY;

	// Sekcja [kernel-buffer2D]
	QString erode_2d;
	QString dilate_2d;
	QString gradient_2d;

	// Sekcja [kernel-buffer1D]
	QString erode_1d;
	QString dilate_1d;
	QString gradient_1d;
	QString subtract_1d;
	QString hitmiss_1d;
	int dataType;

	void saveConfiguration(const QString& filename);
	void loadConfiguration(const QString& filename);
};
