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
	void initHardware(GLDummyWidget* shareWidget);

	void setPreviewImage(const cv::Mat& image, const QSize& maxImgSize);
	void setPreviewImageGL(int w, int h, const QSize& maxImgSize);

	bool useHardware() const { return d_hardware && d_useOpenGL; }

signals:
	void initialized(bool success);

private slots:
	void onGLWidgetInitialized();
	void onGLWidgetError(const QString& msg);

private:
	QVBoxLayout* d_layout;
	GLWidget* d_hardware;
	GLDummyWidget* d_shareWidget;
	QLabel* d_software;
	bool d_useOpenGL;
};

