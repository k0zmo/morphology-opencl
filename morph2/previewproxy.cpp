#include "previewproxy.h"

#include "cvutils.h"

PreviewProxy::PreviewProxy(bool tryOpenGL, QWidget* parent)
	: QWidget(parent)
	, layout(new QVBoxLayout(this))
	, hardware(nullptr)
	, software(nullptr)
	, useOpenGL(tryOpenGL)
{
	if(tryOpenGL)
		initHardware();
	else
		initSoftware();
}

PreviewProxy::~PreviewProxy()
{
}

void PreviewProxy::setPreviewImage(const cv::Mat& image,
								   const cv::Size& maxImgSize)
{
	if(useOpenGL)
	{
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
	software = new QLabel(this);
	software->setText(QString());
	layout->addWidget(software);
}

void PreviewProxy::initHardware()
{
	hardware = new GLWidget(this);
	hardware->resize(1, 1);
	layout->addWidget(hardware);
}
