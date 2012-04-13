#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include "glwidget.h"

class PreviewProxy : public QWidget
{
	Q_OBJECT
public:
	PreviewProxy(bool tryOpenGL, QWidget* parent = 0);
	virtual ~PreviewProxy();

	void setPreviewImage(const cv::Mat& image, const cv::Size &maxImgSize);

private:
	QVBoxLayout* layout;
	GLWidget* hardware;
	QLabel* software;
	bool useOpenGL;

private:
	void initSoftware();
	void initHardware();
};

