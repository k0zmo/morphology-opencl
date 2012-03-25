#include "sepreview.h"
#include "cvutils.h"
#include "controller.h"

#include <QImage>

PreviewLabel::PreviewLabel(QWidget* parent)
	: QLabel(parent)
{
	connect(this, SIGNAL(structuringElementModified(cv::Mat)), 
		gC, SLOT(onStructuringElementModified(cv::Mat)));
}

void PreviewLabel::setPreviewImage(const cv::Mat& se_)
{
	cv::Mat se(se_);
	this->se = se_;

	// Dostosuj go do wyswietlenia
	cv::Size previewSize(width(), height());
	cvu::resizeWithAspect(se, previewSize);

	pixSize = QSizeF((double)se.cols / se_.cols,
		(double)se.rows / se_.rows);

	// Konwersja cv::Mat -> QImage (QPixmap)
	QImage img(reinterpret_cast<const quint8*>(se.data),
		se.cols, se.rows, se.step, 
		QImage::Format_Indexed8);

	// Utworz palete 
	QVector<QRgb> colorTable;
	colorTable << qRgb(0, 0, 0) << qRgb(255, 255, 255);
	img.setColorTable(colorTable);

	setPixmap(QPixmap::fromImage(img));
}

void PreviewLabel::mousePressEvent(QMouseEvent* evt)
{
	Qt::MouseButton btn = evt->button();

	double dx = evt->x() / pixSize.width();
	double dy = evt->y() / pixSize.height();

	int x = static_cast<int>(dx);
	int y = static_cast<int>(dy);

	if(x >= se.cols || y >= se.rows)
		return;

	switch(evt->button())
	{
	case Qt::LeftButton:
		se.at<uchar>(y, x) = 1;
		break;
	case Qt::RightButton:
		se.at<uchar>(y, x) = 0;
		break;
	default: 
		return;
	}

	// poinformuj (controlera) o modyfikacji elementu strukturalnego
	emit structuringElementModified(se);

	setPreviewImage(se);
}

StructuringElementPreview::StructuringElementPreview(QWidget* parent)
	: QDialog(parent)
{
	setupUi(this);
}

void StructuringElementPreview::onStructuringElementChanged(const cv::Mat& se_)
{
	previewLabel->setPreviewImage(se_);
}

void StructuringElementPreview::closeEvent(QCloseEvent* evt)
{
	emit closed();
}