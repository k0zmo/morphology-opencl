#include "previewproxy.h"
#include "cvutils.h"

#include <QMessageBox>

PreviewProxy::PreviewProxy(QWidget *parent)
	: QWidget(parent)
	, layout(new QVBoxLayout(this))
	, hardware(nullptr)
	, software(nullptr)
	, useOpenGL(false)
{
}

PreviewProxy::~PreviewProxy()
{
}

void PreviewProxy::setPreviewImage(const cv::Mat& image,
								   const cv::Size& maxImgSize)
{
	if(useOpenGL)
	{
		if(!hardware)
			return;

		auto coeffs = cvu::scaleCoeffs(image.size(), maxImgSize);
		double fx = coeffs.first;
		double fy = coeffs.second;

		QSize surfaceSize(image.cols * fx, image.rows * fy);

		hardware->setMinimumSize(surfaceSize);
		hardware->setMaximumSize(surfaceSize);
		hardware->setSurface(image);
	}
	else
	{
		if(!software)
			return;

		cv::Mat img(image);
		cvu::fitImageToSize(img, maxImgSize);

		// Konwersja cv::Mat -> QImage -> QPixmap
		QImage qimg(cvu::toQImage(img));
		software->setPixmap(QPixmap::fromImage(qimg));
	}
}

void PreviewProxy::initSoftware()
{
	// Widget wyswietlajacy dany obraz
	useOpenGL = false;
	software = new QLabel(this);
	software->setText(QString());
	layout->addWidget(software);

	emit initialized(true);
}

void PreviewProxy::initHardware()
{
	useOpenGL = true;
	hardware = new GLWidget(this);
	connect(hardware, SIGNAL(error(QString)), SLOT(onGLWidgetError(QString)));
	connect(hardware, SIGNAL(initialized()), SLOT(onGLWidgetInitialized()));
	layout->addWidget(hardware);
	hardware->updateGL();
}

void PreviewProxy::onGLWidgetInitialized()
{
	emit initialized(true);
}

void PreviewProxy::onGLWidgetError(const QString& msg)
{
	QMessageBox::critical(nullptr, "GLWidget critical error",
		msg + "\nSwitching back to software mode.");

	hardware->hide();
	hardware->deleteLater();

	initSoftware();
}
