#pragma once

#include <QWidget>
#include <QVBoxLayout>
#include <QLabel>
#include "glwidget.h"

class PreviewProxy : public QWidget
{
	Q_OBJECT
public:
	PreviewProxy(QWidget* parent = 0);
	virtual ~PreviewProxy();

	void initSoftware();
	void initHardware();

	void setPreviewImage(const cv::Mat& image, const QSize& maxImgSize);
	void setPreviewImageGL(int w, int h, const QSize& maxImgSize);
	GLuint getPreviewImageGL(int w, int h);

signals:
	void initialized(bool success);

private slots:
	void onGLWidgetInitialized();
	void onGLWidgetError(const QString& msg);

private:
	QVBoxLayout* layout;
	GLWidget* hardware;
	QLabel* software;
	bool useOpenGL;
};

