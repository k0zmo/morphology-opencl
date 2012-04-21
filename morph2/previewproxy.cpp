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
								   const QSize& maxImgSize)
{
	if(useOpenGL)
	{
		if(!hardware)
			return;

		cv::Size ms(maxImgSize.width(), maxImgSize.height());
		auto coeffs = cvu::scaleCoeffs(image.size(), ms);

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
		cv::Size ms(maxImgSize.width(), maxImgSize.height());

		cvu::fitImageToSize(img, ms);

		// Konwersja cv::Mat -> QImage -> QPixmap
		QImage qimg(cvu::toQImage(img));

		//QSize surfaceSize(maxImgSize.width, maxImgSize.height);
		//software->setMinimumSize(surfaceSize);
		//software->setMaximumSize(surfaceSize);
		software->setPixmap(QPixmap::fromImage(qimg));
	}
}

void PreviewProxy::setPreviewImageGL(int w, int h, const QSize& maxImgSize)
{
	if(!useOpenGL || !hardware)
		return;

	QSize surfaceSize(w, h);

	if(h > maxImgSize.height() || w > maxImgSize.width())
	{
		double fx;
		if(h > w)
			fx = static_cast<double>(maxImgSize.height()) / h;
		else
			fx = static_cast<double>(maxImgSize.width()) / w;

		surfaceSize.setWidth(w * fx);
		surfaceSize.setHeight(h * fx);
	}

	hardware->setMinimumSize(surfaceSize);
	hardware->setMaximumSize(surfaceSize);
	hardware->updateGL();
}

GLuint PreviewProxy::getPreviewImageGL(int w, int h)
{
	if(!useOpenGL || !hardware)
		return 0;
	return hardware->createEmptySurface(w, h);
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
	hardware->makeCurrent();
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
