#include "previewproxy.h"
#include "cvutils.h"

#include <QMessageBox>

PreviewProxy::PreviewProxy(QWidget *parent)
	: QWidget(parent)
	, d_layout(new QVBoxLayout(this))
	, d_hardware(nullptr)
	, d_shareWidget(nullptr)
	, d_software(nullptr)
	, d_useOpenGL(false)
{
}

PreviewProxy::~PreviewProxy()
{
}

void PreviewProxy::setPreviewImage(const cv::Mat& image,
	const QSize& maxImgSize)
{
	if(d_useOpenGL)
	{
		if(!d_hardware || !d_shareWidget)
			return;

		d_shareWidget->setSurfaceData(image);

		cv::Size ms(maxImgSize.width(), maxImgSize.height());
		auto coeffs = cvu::scaleCoeffs(image.size(), ms);

		double fx = coeffs.first;
		double fy = coeffs.second;

		QSize surfaceSize(image.cols * fx, image.rows * fy);		

		d_hardware->setMinimumSize(surfaceSize);
		d_hardware->setMaximumSize(surfaceSize);
		d_hardware->updateGL();
	}
	else
	{
		if(!d_software)
			return;

		cv::Mat img(image);
		cv::Size ms(maxImgSize.width(), maxImgSize.height());

		cvu::fitImageToSize(img, ms);

		// Konwersja cv::Mat -> QImage -> QPixmap
		QImage qimg(cvu::toQImage(img));

		// QLabel sie sam rozszerzy (chyba)
		//QSize surfaceSize(maxImgSize.width, maxImgSize.height);
		//software->setMinimumSize(surfaceSize);
		//software->setMaximumSize(surfaceSize);

		d_software->setPixmap(QPixmap::fromImage(qimg));
	}
}

void PreviewProxy::setPreviewImageGL(int w, int h, const QSize& maxImgSize)
{
	if(!d_useOpenGL || !d_hardware)
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

	d_hardware->setMinimumSize(surfaceSize);
	d_hardware->setMaximumSize(surfaceSize);
	d_hardware->updateGL();
}

void PreviewProxy::initSoftware()
{
	// Widget wyswietlajacy dany obraz
	d_useOpenGL = false;
	d_software = new QLabel(this);
	d_software->setText(QString());
	d_layout->addWidget(d_software);

	emit initialized(true);
}

void PreviewProxy::initHardware(GLDummyWidget* shareWidget)
{
	d_useOpenGL = true;
	d_hardware = new GLWidget(this, shareWidget);
	d_hardware->setSurface(shareWidget->surface());
	d_shareWidget = shareWidget;

	connect(d_hardware, SIGNAL(error(QString)),
		SLOT(onGLWidgetError(QString)));
	connect(d_hardware, SIGNAL(initialized()),
		SLOT(onGLWidgetInitialized()));

 	d_layout->addWidget(d_hardware);
	d_hardware->updateGL();
}

void PreviewProxy::onGLWidgetInitialized()
{
	emit initialized(true);
}

void PreviewProxy::onGLWidgetError(const QString& msg)
{
	QMessageBox::critical(nullptr, "GLWidget critical error",
		msg + "\nSwitching back to software mode.");

	d_hardware->hide();

	// crashuje
	//d_hardware->deleteLater();

	initSoftware();
}
